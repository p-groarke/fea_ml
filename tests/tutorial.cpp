#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

#include <fea/utils/file.hpp>
#include <gtest/gtest.h>

#pragma warning(push)
#pragma warning(disable : 4702)
#include <torch/torch.h>
#pragma warning(pop)

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>

extern const char* argv0;

namespace {
struct net : torch::nn::Module {
	net(int64_t n, int64_t m)
			: W(register_parameter("W", torch::randn({ n, m })))
			, b(register_parameter("b", torch::randn(m))) {
	}

	torch::Tensor forward(torch::Tensor input) {
		return torch::addmm(b, input, W);
	}

	torch::Tensor W, b;
};

struct net2Impl : torch::nn::Module {
	net2Impl(int64_t n, int64_t m)
			: linear(register_module("linear", torch::nn::Linear(n, m)))
			, another_bias(register_parameter("b", torch::randn(m))) {
	}

	torch::Tensor forward(torch::Tensor input) {
		return linear(input) + another_bias;
	}

	torch::nn::Linear linear;
	torch::Tensor another_bias;
};
TORCH_MODULE(net2);

TEST(torch, tutorial_intro) {
	{
		// torch::Tensor tensor = torch::rand({ 2, 3 });
		// std::cout << tensor << std::endl;
	}

	{
		// torch::tensor = torch::eye(3);
		// std::cout << tensor << std::endl;
	}

	{
		// net n{ 4, 5 };
		// for (const auto& p : n.parameters()) {
		//	std::cout << p << std::endl;
		// }
	}

	{
		net2 n2{ 4, 5 };
		// for (const auto& p : n2->parameters()) {
		//	std::cout << p << std::endl;
		// }
		// std::cout << std::endl;

		for (const auto& p : n2->named_parameters()) {
			std::cout << p.key() << ": " << p.value() << std::endl;
		}
		std::cout << std::endl;

		std::cout << n2->forward(torch::ones({ 2, 4 })) << std::endl;
		std::cout << std::endl;
	}
}

struct dcgan_generatorImpl : torch::nn::Module {
	dcgan_generatorImpl(size_t noise_size)
			: conv1(torch::nn::ConvTranspose2dOptions(noise_size, 256, 4)
							.bias(false))
			, conv2(torch::nn::ConvTranspose2dOptions(256, 128, 3)
							  .stride(2)
							  .padding(1)
							  .bias(false))
			, conv3(torch::nn::ConvTranspose2dOptions(128, 64, 4)
							  .stride(2)
							  .padding(1)
							  .bias(false))
			, conv4(torch::nn::ConvTranspose2dOptions(64, 1, 4)
							  .stride(2)
							  .padding(1)
							  .bias(false))
			, batch_norm1(256)
			, batch_norm2(128)
			, batch_norm3(64) {
		register_module("conv1", conv1);
		register_module("conv2", conv2);
		register_module("conv3", conv3);
		register_module("conv4", conv4);
		register_module("batch_norm1", batch_norm1);
		register_module("batch_norm2", batch_norm2);
		register_module("batch_norm3", batch_norm3);
	}

	torch::Tensor forward(torch::Tensor x) {
		x = torch::relu(batch_norm1(conv1(x)));
		x = torch::relu(batch_norm2(conv2(x)));
		x = torch::relu(batch_norm3(conv3(x)));
		x = torch::tanh(conv4(x));
		return x;
	}

	torch::nn::ConvTranspose2d conv1, conv2, conv3, conv4;
	torch::nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3;
};
TORCH_MODULE(dcgan_generator);

#pragma optimize("", off)
void sample_to_png(const std::filesystem::path& checkpoint_path,
		const std::filesystem::path& out_dir) {
	const std::filesystem::path out_path = out_dir / "samples.png";

	torch::Tensor samples;
	torch::load(samples, checkpoint_path.string());
	// std::cout << samples.size(0) << std::endl;

	int64_t sample_count = samples.size(0);
	constexpr int64_t width = 28;
	constexpr int64_t height = 28;

	// Assume all samples same length.
	std::vector<uint8_t> all_pixels(sample_count * width * height);

	int total_width = width;
	int total_height = height * sample_count;
	for (int64_t i = 0; i < sample_count; ++i) {
		torch::Tensor image = samples[i]
									  .detach()
									  .cpu()
									  .reshape({ width, height })
									  .mul(255)
									  .to(torch::kUInt8);

		auto acc = image.accessor<uint8_t, 2>();
		// std::cout << read.sizes() << std::endl;

		// Insert vertically.
		size_t insert_idx = i * width * height;
		auto insert_it = all_pixels.begin() + insert_idx;
		std::copy(acc.data(), acc.data() + width * height, insert_it);
	}

	assert(all_pixels.size() == total_width * total_height);
	if (!stbi_write_png(out_path.string().c_str(), total_width, total_height, 1,
				all_pixels.data(), total_width * 1)) {

		std::fprintf(stderr, "Write png error : %s\n", stbi_failure_reason());
	}
}
#pragma optimize("", on)

template <class T>
void print_dataset(T& data_loader) {
	for (torch::data::Example<>& batch : *data_loader) {
		std::cout << "Batch size : " << batch.data.size(0) << " | Labels : ";
		for (int64_t i = 0; i < batch.data.size(0); ++i) {
			std::cout << batch.target[i].item<int64_t>() << " ";
		}
		std::cout << std::endl;
	}
}

// #pragma optimize("", off)
TEST(torch, tutorial_dcgan) {
	constexpr int64_t k_noise_size = 100;
	constexpr int64_t k_batch_size = 64;
	constexpr int64_t k_epoch_count = 30;
	constexpr int64_t k_sample_count = 30;
	// constexpr int64_t k_checkpoint_every = 1000;
	constexpr bool k_restore_checkpoint = true;

	const std::filesystem::path exe_dir = fea::executable_dir(argv0);
	const std::filesystem::path mnist_dir = exe_dir / "tests_data" / "mnist";
	const std::filesystem::path checkpoint_dir = exe_dir / "checkpoints";
	if (!std::filesystem::exists(checkpoint_dir)) {
		std::filesystem::create_directories(checkpoint_dir);
	}

	const std::filesystem::path k_generator_checkpoint_name
			= checkpoint_dir / "generator-checkpoint.pt";
	const std::filesystem::path k_g_optimizer_checkpoint_name
			= checkpoint_dir / "generator-optimizer-checkpoint.pt";
	const std::filesystem::path k_discriminator_checkpoint_name
			= checkpoint_dir / "discriminator-checkpoint.pt";
	const std::filesystem::path k_d_optimizer_checkpoint_name
			= checkpoint_dir / "discriminator-optimizer-checkpoint.pt";
	const std::filesystem::path k_sample_checkpoint_name
			= checkpoint_dir / "dcgan-sample-checkpoint.pt";

	torch::Device device(
			torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
	if (!torch::cuda::is_available()) {
		std::cout << "CUDA not found, running on CPU!" << std::endl;
	}

	dcgan_generator generator{ k_noise_size };
	generator->to(device);

	torch::nn::Sequential discriminator{
		// layer 1
		torch::nn::Conv2d{
				torch::nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(
						false),
		},
		torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
		// layer 2
		torch::nn::Conv2d{
				torch::nn::Conv2dOptions(64, 128, 4)
						.stride(2)
						.padding(1)
						.bias(false),
		},
		torch::nn::BatchNorm2d(128),
		torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
		// layer 3
		torch::nn::Conv2d{
				torch::nn::Conv2dOptions(128, 256, 4)
						.stride(2)
						.padding(1)
						.bias(false),
		},
		torch::nn::BatchNorm2d(256),
		torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
		// layer 4
		torch::nn::Conv2d{
				torch::nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).bias(
						false),
		},
		torch::nn::Sigmoid{},
	};
	discriminator->to(device);

	// Load dataset.
	auto dataset = torch::data::datasets::MNIST(mnist_dir.string())
						   .map(torch::data::transforms::Normalize<>(0.5, 0.5))
						   .map(torch::data::transforms::Stack<>());

	const int64_t batches_per_epoch
			= std::ceil(dataset.size().value() / double(k_batch_size));

	auto data_loader = torch::data::make_data_loader(std::move(dataset),
			torch::data::DataLoaderOptions{}
					.batch_size(k_batch_size)
					.workers(2));

	// Optimizers, whatever they do.
	torch::optim::Adam g_optimizer{
		generator->parameters(),
		torch::optim::AdamOptions(2e-4).betas({ 0.5, 0.5 }),
	};

	torch::optim::Adam d_optimizer{
		discriminator->parameters(),
		torch::optim::AdamOptions(5e-4).betas({ 0.5, 0.5 }),
	};

	if (k_restore_checkpoint) {
		torch::load(generator, k_generator_checkpoint_name.string());
		torch::load(g_optimizer, k_g_optimizer_checkpoint_name.string());
		torch::load(discriminator, k_discriminator_checkpoint_name.string());
		torch::load(d_optimizer, k_d_optimizer_checkpoint_name.string());
	}


	for (int64_t epoch = 1; epoch <= k_epoch_count; ++epoch) {
		int64_t batch_index = 0;
		for (torch::data::Example<>& batch : *data_loader) {
			// const uint64_t batch_size = batch.data.size(0);

			// Train discriminator with real images.
			discriminator->zero_grad();
			torch::Tensor real_images = batch.data.to(device);
			torch::Tensor real_labels = torch::empty(batch.data.size(0), device)
												.uniform_(0.8, 1.0);
			torch::Tensor real_output = discriminator->forward(real_images)
												.reshape(real_labels.sizes());
			torch::Tensor d_loss_real
					= torch::binary_cross_entropy(real_output, real_labels);
			d_loss_real.backward();

			// Train discriminator with fake images.
			torch::Tensor noise = torch::randn(
					{ batch.data.size(0), k_noise_size, 1, 1 }, device);
			torch::Tensor fake_images = generator->forward(noise);
			torch::Tensor fake_labels
					= torch::zeros(batch.data.size(0), device);
			torch::Tensor fake_output
					= discriminator->forward(fake_images.detach())
							  .reshape(fake_labels.sizes());
			torch::Tensor d_loss_fake
					= torch::binary_cross_entropy(fake_output, fake_labels);
			d_loss_fake.backward();

			torch::Tensor d_loss = d_loss_real + d_loss_fake;
			d_optimizer.step();

			// Train generator.
			generator->zero_grad();
			fake_labels.fill_(1);
			fake_output = discriminator->forward(fake_images)
								  .reshape(fake_labels.sizes());
			torch::Tensor g_loss
					= torch::binary_cross_entropy(fake_output, fake_labels);
			g_loss.backward();
			g_optimizer.step();

			++batch_index;
			std::printf(
					"\r[%2lld/%2lld][%3lld/%3lld] D_loss: %.4f | G_loss: %.4f",
					epoch, k_epoch_count, batch_index, batches_per_epoch,
					d_loss.item<float>(), g_loss.item<float>());
		}


		// Checkpoint the model and optimizer state.
		torch::save(generator, k_generator_checkpoint_name.string());
		torch::save(g_optimizer, k_g_optimizer_checkpoint_name.string());
		torch::save(discriminator, k_discriminator_checkpoint_name.string());
		torch::save(d_optimizer, k_d_optimizer_checkpoint_name.string());

		// Sample the generator and save the images.
		torch::Tensor samples = generator->forward(
				torch::randn({ k_sample_count, k_noise_size, 1, 1 }, device));
		torch::save((samples + 1.0) / 2.0, k_sample_checkpoint_name.string());

		std::cout << std::endl << "-> checkpoint " << epoch << std::endl;
	}

	// Saves samples to png.
	sample_to_png(k_sample_checkpoint_name, checkpoint_dir);
}
// #pragma optimize("", on)
} // namespace