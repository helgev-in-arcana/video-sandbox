//! rsmpeg を閉じ込めた I/O 境界。デコード（ソース Iterator）とエンコード（シンク）。
//! ここだけが FFmpeg に依存する。内部の `Frame` は常に RGBA8。

use std::ffi::CString;
use std::ptr::copy_nonoverlapping;

use anyhow::{anyhow, bail, Result};
use rsmpeg::avcodec::{AVCodec, AVCodecContext};
use rsmpeg::avformat::{AVFormatContextInput, AVFormatContextOutput};
use rsmpeg::avutil::{ra, AVFrame, AVHWDeviceContext};
use rsmpeg::error::RsmpegError;
use rsmpeg::ffi;
use rsmpeg::swscale::SwsContext;

use crate::frame::Frame;
use crate::pixel::Pixel;

/// 行ごとに linesize 分の stride を詰めて、パック済みの [`Pixel`] 列を作る。
fn pack_rgba(plane: *const u8, stride: usize, width: usize, height: usize) -> Vec<Pixel> {
    let mut data = vec![Pixel::default(); width * height];
    let dst: &mut [u8] = bytemuck::cast_slice_mut(&mut data);
    let row = width * 4;
    for y in 0..height {
        unsafe {
            copy_nonoverlapping(plane.add(y * stride), dst.as_mut_ptr().add(y * row), row);
        }
    }
    data
}

/// デコーダのフォーマット選択コールバック。候補に HW フォーマット（D3D11 / DXVA2）が
/// あればそれを選び、無ければ先頭（ソフト）にフォールバックする。
unsafe extern "C" fn get_format_hw(
    _ctx: *mut ffi::AVCodecContext,
    fmts: *const ffi::AVPixelFormat,
) -> ffi::AVPixelFormat {
    unsafe {
        let first = *fmts;
        let mut p = fmts;
        while *p != ffi::AV_PIX_FMT_NONE {
            if *p == ffi::AV_PIX_FMT_D3D11 || *p == ffi::AV_PIX_FMT_DXVA2_VLD {
                return *p;
            }
            p = p.add(1);
        }
        first
    }
}

// ---------------------------------------------------------------------------
// デコード（ソース）
// ---------------------------------------------------------------------------

/// 入力を開き、各フレームを RGBA8 化して yield する Iterator を返す。
/// 開けなかった場合は警告を出して空 Iterator を返す（ソースは無謬な Iterator）。
pub fn decode_iter(path: &str, hwaccel: Option<&str>) -> Box<dyn Iterator<Item = Frame> + Send> {
    match Decoder::open(path, hwaccel) {
        Ok(d) => Box::new(d),
        Err(e) => {
            eprintln!("[video-sandbox] 入力 '{path}' を開けませんでした: {e:#}");
            Box::new(std::iter::empty())
        }
    }
}

struct Decoder {
    input: AVFormatContextInput,
    decode_ctx: AVCodecContext,
    stream_index: usize,
    sws: Option<SwsContext>,
    flushed: bool,
}

impl Decoder {
    fn open(path: &str, hwaccel: Option<&str>) -> Result<Self> {
        let path_c = CString::new(path)?;
        let input = AVFormatContextInput::open(&path_c)?;
        let (stream_index, decoder) = input
            .find_best_stream(ffi::AVMEDIA_TYPE_VIDEO)?
            .ok_or_else(|| anyhow!("動画ストリームが見つかりません"))?;
        let mut decode_ctx = AVCodecContext::new(&decoder);
        {
            let stream = &input.streams()[stream_index];
            decode_ctx.apply_codecpar(&stream.codecpar())?;
        }

        // HW デコードの選択（[`DecodeSettings::hwaccel`]）。例: d3d11va（AMD/Windows 汎用）/ dxva2。
        // フレームは GPU サーフェスで返るので、next() で hwframe_transfer_data によりシステム
        // メモリへ落としてから sws→RGBA に流す。
        if let Some(kind) = hwaccel.filter(|s| !s.is_empty()) {
            let dev_type = match kind {
                "d3d11va" => ffi::AV_HWDEVICE_TYPE_D3D11VA,
                "dxva2" => ffi::AV_HWDEVICE_TYPE_DXVA2,
                other => bail!("未知の VS_HWDEC: {other}（d3d11va / dxva2 のみ）"),
            };
            let hwdev = AVHWDeviceContext::create(dev_type, None, None, 0)?;
            decode_ctx.set_hw_device_ctx(hwdev);
            decode_ctx.set_get_format(Some(get_format_hw));
            eprintln!("[dec] HW デコード: {kind}");
        }

        decode_ctx.open(None)?;
        Ok(Decoder { input, decode_ctx, stream_index, sws: None, flushed: false })
    }

    /// HW サーフェスなら GPU からシステムメモリへ転送する。ソフトフレームはそのまま返す。
    fn maybe_download(&self, frame: AVFrame) -> AVFrame {
        if frame.hw_frames_ctx.is_null() {
            return frame; // 既にシステムメモリ上（ソフトデコード）
        }
        let mut sw = AVFrame::new();
        sw.hwframe_transfer_data(&frame)
            .expect("hwframe_transfer_data（GPU→システムメモリ）に失敗");
        // 転送はピクセルデータのみ。タイムスタンプは引き継がれないので手で写す。
        sw.set_pts(frame.pts);
        sw
    }

    /// デコード済み AVFrame（YUV 等）を RGBA8 の `Frame` に変換する。
    fn convert(&mut self, src: &AVFrame) -> Frame {
        let w = src.width;
        let h = src.height;
        if self.sws.is_none() {
            self.sws = Some(
                SwsContext::get_context(
                    w,
                    h,
                    src.format,
                    w,
                    h,
                    ffi::AV_PIX_FMT_RGBA,
                    ffi::SWS_BILINEAR,
                    None,
                    None,
                    None,
                )
                .expect("RGBA への SwsContext を作成できませんでした"),
            );
        }
        let mut dst = AVFrame::new();
        dst.set_width(w);
        dst.set_height(h);
        dst.set_format(ffi::AV_PIX_FMT_RGBA);
        dst.get_buffer(0).expect("RGBA バッファ確保失敗");
        self.sws
            .as_mut()
            .unwrap()
            .scale_frame(src, 0, h, &mut dst)
            .expect("sws_scale (decode) 失敗");

        let data = pack_rgba(dst.data[0], dst.linesize[0] as usize, w as usize, h as usize);
        Frame::from_rgba(w as u32, h as u32, data, src.pts)
    }
}

impl Iterator for Decoder {
    type Item = Frame;

    fn next(&mut self) -> Option<Frame> {
        loop {
            match self.decode_ctx.receive_frame() {
                Ok(frame) => {
                    let frame = self.maybe_download(frame);
                    return Some(self.convert(&frame));
                }
                Err(RsmpegError::DecoderDrainError) => {
                    if self.flushed {
                        return None;
                    }
                    // デコーダがフレームを欲しがっている。次のビデオパケットを供給。
                    loop {
                        match self.input.read_packet() {
                            Ok(Some(pkt)) => {
                                if pkt.stream_index as usize != self.stream_index {
                                    continue;
                                }
                                if let Err(e) = self.decode_ctx.send_packet(Some(&pkt)) {
                                    eprintln!("[video-sandbox] send_packet: {e}");
                                    return None;
                                }
                                break;
                            }
                            Ok(None) => {
                                // 入力 EOF → デコーダを flush（null パケット送出）。
                                let _ = self.decode_ctx.send_packet(None);
                                self.flushed = true;
                                break;
                            }
                            Err(e) => {
                                eprintln!("[video-sandbox] read_packet: {e}");
                                return None;
                            }
                        }
                    }
                }
                Err(RsmpegError::DecoderFlushedError) => return None,
                Err(e) => {
                    eprintln!("[video-sandbox] receive_frame: {e}");
                    return None;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// エンコード（シンク）
// ---------------------------------------------------------------------------

use crate::pipeline::EncodeSettings;

pub struct Encoder {
    output: AVFormatContextOutput,
    encode_ctx: AVCodecContext,
    sws: Option<SwsContext>,
    stream_index: i32,
    stream_time_base: ffi::AVRational,
    width: i32,
    height: i32,
    next_pts: i64,
}

impl Encoder {
    pub fn new(path: &str, width: u32, height: u32, settings: &EncodeSettings) -> Result<Self> {
        let w = width as i32;
        let h = height as i32;
        let path_c = CString::new(path)?;
        let mut output = AVFormatContextOutput::create(&path_c)?;

        // 使用エンコーダ（[`EncodeSettings::encoder`]）。既定は libx264（ソフト）。
        // 例: h264_amf（AMD HW）/ h264_nvenc / h264_qsv。
        let enc_name = settings.encoder.as_deref().unwrap_or("libx264");
        let enc_name_c = CString::new(enc_name)?;
        let encoder = AVCodec::find_encoder_by_name(&enc_name_c)
            .ok_or_else(|| anyhow!("エンコーダ '{enc_name}' が見つかりません"))?;
        eprintln!("[enc] エンコーダ: {enc_name}");
        let mut encode_ctx = AVCodecContext::new(&encoder);
        encode_ctx.set_width(w);
        encode_ctx.set_height(h);
        encode_ctx.set_pix_fmt(ffi::AV_PIX_FMT_YUV420P);
        let tb = ra(1, settings.fps);
        encode_ctx.set_time_base(tb);
        encode_ctx.set_framerate(ra(settings.fps, 1));
        encode_ctx.set_bit_rate(settings.bit_rate);
        encode_ctx.set_gop_size(12);

        // コンテナがグローバルヘッダを要求する場合（mp4 等）はフラグを立てる。
        if output.oformat().flags & ffi::AVFMT_GLOBALHEADER as i32 != 0 {
            encode_ctx.set_flags(encode_ctx.flags | ffi::AV_CODEC_FLAG_GLOBAL_HEADER as i32);
        }
        encode_ctx.open(None)?;

        let stream_index;
        {
            let mut stream = output.new_stream();
            stream.set_codecpar(encode_ctx.extract_codecpar());
            stream.set_time_base(tb);
            stream_index = stream.index;
        }
        output.write_header(&mut None)?;
        let stream_time_base = output.streams()[stream_index as usize].time_base;

        Ok(Encoder {
            output,
            encode_ctx,
            sws: None,
            stream_index,
            stream_time_base,
            width: w,
            height: h,
            next_pts: 0,
        })
    }

    pub fn encode(&mut self, frame: Frame) -> Result<()> {
        // RGBA8（パック済み）→ AVFrame(RGBA) へ詰める。
        let mut src = AVFrame::new();
        src.set_width(self.width);
        src.set_height(self.height);
        src.set_format(ffi::AV_PIX_FMT_RGBA);
        src.get_buffer(0)?;
        {
            let stride = src.linesize[0] as usize;
            let row = self.width as usize * 4;
            let plane = src.data[0];
            for y in 0..self.height as usize {
                unsafe {
                    copy_nonoverlapping(
                        frame.data().as_ptr().add(y * row),
                        plane.add(y * stride),
                        row,
                    );
                }
            }
        }

        if self.sws.is_none() {
            self.sws = Some(
                SwsContext::get_context(
                    self.width,
                    self.height,
                    ffi::AV_PIX_FMT_RGBA,
                    self.width,
                    self.height,
                    ffi::AV_PIX_FMT_YUV420P,
                    ffi::SWS_BILINEAR,
                    None,
                    None,
                    None,
                )
                .ok_or_else(|| anyhow!("yuv420p への SwsContext を作成できませんでした"))?,
            );
        }
        let mut dst = AVFrame::new();
        dst.set_width(self.width);
        dst.set_height(self.height);
        dst.set_format(ffi::AV_PIX_FMT_YUV420P);
        dst.get_buffer(0)?;
        self.sws.as_mut().unwrap().scale_frame(&src, 0, self.height, &mut dst)?;

        // 出力 pts はフレーム番号で振り直す（エンコーダ time_base 基準）。
        dst.set_pts(self.next_pts);
        self.next_pts += 1;

        self.drain(Some(&dst))
    }

    pub fn finish(mut self) -> Result<()> {
        self.drain(None)?;
        self.output.write_trailer()?;
        Ok(())
    }

    /// エンコーダにフレームを送り、出たパケットを mux する。`None` で flush。
    fn drain(&mut self, frame: Option<&AVFrame>) -> Result<()> {
        self.encode_ctx.send_frame(frame)?;
        loop {
            let mut pkt = match self.encode_ctx.receive_packet() {
                Ok(p) => p,
                Err(RsmpegError::EncoderDrainError) | Err(RsmpegError::EncoderFlushedError) => {
                    break;
                }
                Err(e) => return Err(e.into()),
            };
            pkt.set_stream_index(self.stream_index);
            // 各パケットに 1 フレーム分の duration を持たせる（エンコーダ time_base 単位）。
            // これが無いと movenc が末尾フレームの長さを決められず discard フラグを立て、
            // デコード時に 1 枚欠ける。rescale_ts は duration も変換する。
            pkt.set_duration(1);
            pkt.rescale_ts(self.encode_ctx.time_base, self.stream_time_base);
            self.output.interleaved_write_frame(&mut pkt)?;
        }
        Ok(())
    }
}
