import argparse
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

transform = transforms.Compose([
	transforms.ToTensor(),  # convert the image to a Tensor
	transforms.Normalize(mean=[0.485, 0.456, 0.406],
	                     std=[0.229, 0.224, 0.225])
])


def test(test_path, model_weight_path):
	test_dataset = datasets.ImageFolder(root=test_path,
	                                    transform=transform)
	test_load = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model = torch.jit.load(model_weight_path, map_location=device)
	model.eval()

	for i, (inputs, labels) in enumerate(test_load):
		inputs = inputs.to(device)
		labels = labels.to(device)

		outputs = model(inputs)
		# Record the correct predictions for training data
		_, predicted = torch.max(outputs, 1)
		print("i: " + str(i) + " | Label: " + str(labels) + " | Predicted: " + str(predicted))


if __name__ == '__main__':

	# Create the parser and add arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--test-path',default='../test/', help="test main dir")
	parser.add_argument("--model-weight-path",default='HE_mitosis.pt', help="pt file")

	# Parse and print the results
	args = parser.parse_args()
	test(test_path=args.test_path, model_weight_path=args.model_weight_path)