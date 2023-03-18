#include <gtest/gtest.h>
#include <iostream>

#pragma warning(push)
#pragma warning(disable : 4702)
#include <torch/torch.h>
#pragma warning(pop)

TEST(fea_ml, tutorial) {
	torch::Tensor tensor = torch::rand({ 2, 3 });
	std::cout << tensor << std::endl;
}