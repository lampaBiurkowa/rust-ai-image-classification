use std::path::Path;

use batcher::{ImageClassificationBatch, ImageClassificationBatcher};
use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu}, data::{dataloader::DataLoaderBuilder, dataset::vision::{Annotation, ImageDatasetItem, ImageFolderDataset}}, nn::{conv::{Conv2d, Conv2dConfig}, Linear, LinearConfig, PaddingConfig2d}, optim::{decay::WeightDecayConfig, AdamConfig}, prelude::*, record::{FullPrecisionSettings, NamedMpkFileRecorder}, tensor::{activation::relu, backend::AutodiffBackend, module::max_pool2d}, train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep}
};
use image_classification::{batcher, utils};
use nn::{loss::CrossEntropyLossConfig, pool::MaxPool2dConfig, Dropout, DropoutConfig};
use utils::{basic_train, create_artifact_dir};
use burn::data::dataloader::batcher::Batcher;
use burn::record::Recorder;
use burn::data::dataloader::Dataset;

#[derive(Config)]
pub struct ModelConfig {
    #[config(default = "6")]
    num_classes: usize,
    #[config(default = "16")]
    channels: usize,
    #[config(default = "128")]
    hidden_size: usize,
    #[config(default = "0.2")]
    dropout: f64,
    #[config(default = "3")]
    kernel_size: usize,
    #[config(default = "1")]
    padding: usize,
    #[config(default = "1")]
    stride: usize,
    #[config(default = "256")]
    image_height: usize,
    #[config(default = "256")]
    image_width: usize,
    max_pool: MaxPool2dConfig,
}

#[derive(Config)]
struct TrainingConfig {
    #[config(default = "1")]
    seed: u64,
    #[config(default = "8")]
    batch_size: usize,
    #[config(default = "4")]
    num_workers: usize,
    #[config(default = "0.001")]
    learning_rate: f64,
    #[config(default = "2")]
    num_epochs: usize,
    artifacts_dir: String,
    model_name: String,
    optimizer: AdamConfig,
}

impl ModelConfig {
    fn compute_flattened_size(&self) -> usize {
        let (mut height, mut width) = (self.image_height, self.image_width);

        height = (height + 2 * self.padding - self.kernel_size) / self.stride + 1;
        width = (width + 2 * self.padding - self.kernel_size) / self.stride + 1;

        height = (height + 2 * 1 - 2) / 2 + 1;
        width = (width + 2 * 1 - 2) / 2 + 1;

        height = (height + 2 * self.padding - self.kernel_size) / self.stride + 1;
        width = (width + 2 * self.padding - self.kernel_size) / self.stride + 1;

        height = (height + 2 * 1 - 2) / 2 + 1;
        width = (width + 2 * 1 - 2) / 2 + 1;

        self.channels * height * width
    }

    fn init<B: Backend>(&self, device: &B::Device) -> ImageClassificationModel<B> {
        let conv1 = Conv2dConfig::new([3, self.channels], [self.kernel_size, self.kernel_size]) 
            .with_stride([self.stride, self.stride])
            .with_padding(PaddingConfig2d::Explicit(self.padding, self.padding))
            .init(device);

        let conv2 = Conv2dConfig::new([self.channels, self.channels], [self.kernel_size, self.kernel_size])
            .with_stride([self.stride, self.stride])
            .with_padding(PaddingConfig2d::Explicit(self.padding, self.padding))
            .init(device);

        let fc1 = LinearConfig::new(self.compute_flattened_size(), self.hidden_size).init(device);
        let fc2 = LinearConfig::new(self.hidden_size, self.num_classes).init(device);
        let dropout = DropoutConfig::new(self.dropout).init();

        ImageClassificationModel { conv1, conv2, fc1, fc2, dropout }
    }
}

#[derive(Module, Debug)]
struct ImageClassificationModel<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> ImageClassificationModel<B> {
    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv1.forward(input); 
        let x = relu(x);                  
        let x = max_pool2d(x, [2, 2], [2, 2], [1, 1], [1, 1]);

        let x = self.conv2.forward(x); 
        let x = relu(x);              
        let x = max_pool2d(x, [2, 2], [2, 2], [1, 1], [1, 1]);

        let dims = x.dims();
        let num_features = dims[1] * dims[2] * dims[3];
        let x = x.reshape([-1, num_features as i32]);

        let x = self.fc1.forward(x); 
        let x = self.dropout.forward(x);
        let x = relu(x);             
        self.fc2.forward(x)
    }

    fn forward_classification(&self, item: ImageClassificationBatch<B>) -> ClassificationOutput<B> {
        let targets = item.targets;
        let output = self.forward(item.images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<ImageClassificationBatch<B>, ClassificationOutput<B>> for ImageClassificationModel<B> {
    fn step(&self, item: ImageClassificationBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(item);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<ImageClassificationBatch<B>, ClassificationOutput<B>> for ImageClassificationModel<B> {
    fn step(&self, item: ImageClassificationBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(item)
    }
}

fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: ImageDatasetItem) {
    let config = ModelConfig::load(Path::new(artifact_dir).join("model_config.json"))
        .expect("Config file should exist");

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let record = recorder
        .load(
            Path::new(artifact_dir).join("model"),
            &device,
        )
        .expect("Trained model should exist");
    let model = config.init::<B>(&device).load_record(record);

    let label = item.annotation.clone();
    let batcher = ImageClassificationBatcher::new(device.clone());
    let batch = batcher.batch(vec![item]);

    let output = model.forward(batch.images);
    let predicted = output.clone().argmax(1).into_scalar();

    if let Annotation::Label(expected) = label {
        println!(
            "Predicted: {} | Expected: {} | Raw Output: {:?}",
            predicted, expected, output
        );
    } else {
        println!("No label provided for comparison. Predicted: {}", predicted);
    }
}

fn main() {
    type Backend = Autodiff<Wgpu>;
    let device = WgpuDevice::IntegratedGpu(0);

    let model_config = ModelConfig::new(MaxPool2dConfig::new([1, 1]));
    let training_config = TrainingConfig::new(
        "artifacts".to_owned(),
        "model".to_owned(),
        AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5)))
    );

    let model = model_config.init(&device);
    create_artifact_dir(&training_config.artifacts_dir);
    Backend::seed(training_config.seed);

    let batcher_train = ImageClassificationBatcher::<Backend>::new(device.clone());
    let batcher_valid = ImageClassificationBatcher::<Wgpu>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(training_config.batch_size)
        .shuffle(training_config.seed)
        .num_workers(training_config.num_workers)
        .build(ImageFolderDataset::new_classification("../data/final/train").unwrap());

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(training_config.batch_size)
        .shuffle(training_config.seed)
        .num_workers(training_config.num_workers)
        .build(ImageFolderDataset::new_classification("../data/final/valid").unwrap());

    let trained_model = basic_train(
        &training_config.artifacts_dir,
        &device,
        training_config.num_epochs,
        training_config.learning_rate,
        model,
        training_config.optimizer.init(),
        dataloader_train,
        dataloader_valid,
    );

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    trained_model
        .save_file(Path::new(&training_config.artifacts_dir).join(&training_config.model_name), &recorder)
        .expect("Should be able to save the model");

    training_config
        .save(Path::new(&training_config.artifacts_dir).join("training_config.json"))
        .unwrap();

    model_config
        .save(Path::new(&training_config.artifacts_dir).join("model_config.json"))
        .unwrap();

    let dataset = ImageFolderDataset::new_classification("../data/final/valid").unwrap(); // :D/
    if let Some(item) = dataset.get(0) {
        infer::<Backend>(&training_config.artifacts_dir, device, item);
    } else {
        println!("No images found in the validation dataset.");
    }
}