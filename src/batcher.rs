use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::{Annotation, ImageDatasetItem, PixelDepth}},
    prelude::*,
};

#[derive(Clone, Debug)]
pub struct ImageClassificationBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct ImageClassificationBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> ImageClassificationBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<ImageDatasetItem, ImageClassificationBatch<B>> for ImageClassificationBatcher<B> {
    fn batch(&self, items: Vec<ImageDatasetItem>) -> ImageClassificationBatch<B> {
        let images = items
            .iter()
            .map(|x| x.image.iter().map(|x| match x {
                PixelDepth::F32(v) => *v,
                PixelDepth::U16(v) => *v as f32,
                PixelDepth::U8(v) => *v as f32
            }).collect::<Vec<f32>>())
            .map(|x| TensorData::from(x.as_slice()))
            .map(|data| Tensor::<B, 1>::from_floats(data, &self.device))
            .map(|tensor| tensor.reshape([3, 256, 256]))
            .map(|tensor| (tensor / 255))
            .collect();
        
        let targets = items
            .iter()
            .map(|item| {
                let as_usize = match item.annotation {
                    Annotation::Label(x) => x,
                    _ => panic!("xd")
                };
                Tensor::<B, 1, Int>::from_data(
                    TensorData::from([as_usize].as_slice()),
                    &self.device,
                )
            })
            .collect();

        let images = Tensor::stack(images, 0);
        let targets = Tensor::cat(targets, 0);

        ImageClassificationBatch { images, targets }
    }
}