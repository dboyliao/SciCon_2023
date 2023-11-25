convert:
	utensor-cli convert cifar10_cnn.pb --output-nodes=fully_connect_2/logits

cpp: convert
	@mv models/cifar10_cnn/*.{cpp,hpp} .
	@mv constants/cifar10_cnn/*.hpp .
	@rm -rf models constants
