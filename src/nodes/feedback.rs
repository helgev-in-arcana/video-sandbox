use video_pipeline::{Frame, FrameCtx, Process};

#[derive(Clone)]
pub struct Feedback<MX, FW, FB>
where
    MX: FnMut(Frame, Frame, FrameCtx) -> Frame,
    FW: Process,
    FB: Process,
{
    buffer: Option<Frame>,

    mixer: MX,
    feedforward_element: FW,
    feedback_element: FB,
}

impl<MX, FW, FB> Feedback<MX, FW, FB>
where
    MX: FnMut(Frame, Frame, FrameCtx) -> Frame,
    FW: Process,
    FB: Process,
{
    pub fn new(mixer: MX, feedforward_element: FW, feedback_element: FB) -> Self {
        Self {
            buffer: None,
            mixer,
            feedforward_element,
            feedback_element,
        }
    }
}

impl<MX, FW, FB> Process for Feedback<MX, FW, FB>
where
    MX: FnMut(Frame, Frame, FrameCtx) -> Frame,
    FW: Process,
    FB: Process,
{
    fn process(&mut self, frame: Frame, ctx: FrameCtx) -> Frame {
        let buffer = self
            .buffer
            .take()
            .unwrap_or_else(|| frame.clone());
        let mixed = (self.mixer)(frame, buffer, ctx);

        let feedforward = self.feedforward_element.process(mixed, ctx);
        self.buffer = Some(self.feedback_element.process(feedforward.clone(), ctx));

        feedforward
    }
}
