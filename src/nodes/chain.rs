use video_pipeline::{Frame, FrameCtx, Process};

/// 複数段の [`Process`] を連結するノード。
pub struct ChainNode {
    chain: Vec<Box<dyn Process>>,
}

impl ChainNode {
    pub fn new() -> Self {
        Self { chain: Vec::new() }
    }

    pub fn push(&mut self, process: impl Process + 'static) {
        self.chain.push(Box::new(process));
    }
}

impl Process for ChainNode {
    fn process(&mut self, frame: Frame, ctx: FrameCtx) -> Frame {
        let mut frame = frame;
        for process in &mut self.chain {
            frame = process.process(frame, ctx);
        }
        frame
    }
}
