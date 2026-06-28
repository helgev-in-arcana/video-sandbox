use std::sync::Arc;

use video_pipeline::{Frame, FrameCtx, Pixel, Process};

#[derive(Clone)]
pub struct PixelSort {
    mask_f: Arc<dyn Fn(Pixel) -> bool + Send + Sync>,
    sort_f: Arc<dyn Fn(&mut [Pixel]) + Send + Sync>,
}

impl Process for PixelSort {
    fn process(&mut self, mut frame: Frame, ctx: FrameCtx) -> Frame {
        frame.per_iter_row(&ctx, |ctx, y, row| {
            let mut sort_start = 0;
            let mut sort_end = 0;

            for i in 0..row.len() {
                if (self.mask_f)(row[i]) {
                    sort_end = i;
                } else if sort_start == sort_end {
                    sort_start = i;
                    sort_end = i;
                } else {
                    (self.sort_f)(&mut row[sort_start..=sort_end]);
                    sort_start = i;
                    sort_end = i;
                }
            }
        });

        frame
    }
}

impl PixelSort {
    pub fn simple(
        pixel_value: impl Fn(Pixel) -> f32 + Send + Sync + Clone + 'static,
        threshold: f32,
    ) -> Self {
        let pixel_value_clone = pixel_value.clone();
        Self {
            mask_f: Arc::new(move |p| (pixel_value_clone)(p) > threshold),
            sort_f: Arc::new(move |pixels| {
                pixels.sort_by(|a, b| {
                    let a_val = (pixel_value)(*a);
                    let b_val = (pixel_value)(*b);
                    a_val
                        .partial_cmp(&b_val)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            }),
        }
    }

    // pub fn simple_with_time

    pub fn distorted_and_sort(
        sort_mask: impl Fn(Pixel) -> bool + Send + Sync + 'static,
        sort_value: impl Fn(Pixel) -> f32 + Send + Sync + 'static,
    ) -> Self {
        Self {
            mask_f: Arc::new(sort_mask),
            sort_f: Arc::new(move |pixels| {
                pixels.sort_by(|a, b| {
                    let a_val = (sort_value)(*a);
                    let b_val = (sort_value)(*b);
                    a_val
                        .partial_cmp(&b_val)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            }),
        }
    }

    pub fn custom(
        mask_f: impl Fn(Pixel) -> bool + Send + Sync + 'static,
        sort_f: impl Fn(&mut [Pixel]) + Send + Sync + 'static,
    ) -> Self {
        Self {
            mask_f: Arc::new(mask_f),
            sort_f: Arc::new(sort_f),
        }
    }
}
