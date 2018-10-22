# Data‐driven identification and control of nonlinear systems using multiple NARMA‐L2 models
This repository provides homemade C++ code for a fixed-structure feedforward neural network to identify (approximate) a NARMA-L2 model using experimental input-output data.  For more details about NARMA model, NARMA-L2 model, and multiple NARMA-L2 model based control, please refer to our paper [Data‐driven identification and control of nonlinear systems using multiple NARMA‐L2 models](https://onlinelibrary.wiley.com/doi/abs/10.1002/rnc.3818). Th network structure for a NARMA-L2 model is shown below.
![NARMA-L2](/doc/img/NARMA-L2.png)

*Note that this project is mainly to practice the development of neural networks using a difficult language like C++ from scratch.  In our paper, we actually build the network using MATLAB neural network toolbox, which is much more sophisticated, though less efficient, than this C++ based implementation. The fitting performance of MATLAB toolbox is better than this naïve implementation due to its more advanced training algorithms. If you want industry-level network tools, then PyTorch or TensorFlow are recommended.*

## Features

This neural network is coded in C++ from scratch instead of depending on existent libraries, such as MATLAB neural network toolbox or *TensorFlow*, to make it a light-weighted, and self-contained tool specially designed for NARMA-L2 model identification and control. 
- This implementation is very efficient due to **full vectorization** by properly using the [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) linear algebra library.
- The batch size for network training can be specified arbitrarily to support full-batch, mini-batch and totally stochastic gradient network training.
- The hidden layer supports both ReLU and Tanh activations. 
- An arbitrary number of hidden layers can be specified though for system identification usually one or two hidden layers are enough. The two subnets shown above, N1 and N2, can use different number of hidden layers and hidden neurons.
- Training supports both naïve stochastic gradient descent and gradient descent with momentum. 

## How to use

- This project depends on  [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) for matrix computation. Please first download the header-only library  [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) and add it to the including file search path of your C++ compiler. 
- A C++ compiler supporting the C++ 11 standard is required. 
- This project has been developed and tested using Visual Studio 2017 on Windows 10.

# Documentation

All the files are documented in details using *Doxygen*-style comments. A one-page [PDF](/doc/NARMA-L2-network-model-implementation.pdf) is also provided to list the necessary formulas to help understand the implementation.

# Example

An example system is given in the  [PDF](/doc/NARMA-L2-network-model-implementation.pdf).  The main function in the [main.cpp](/NARMA-L2/core/main.cpp) demonstrates how to approximate this system using a NARMA-L2 model network. Note that this system itself is always in NARMA-L2 form. For general systems, multiple NARMA-L2 models are needed. Please refer to our paper for details.

```cpp
int main()
{
	// get training set
	MatrixXd X;
	VectorXd u, y;
	std::tie(X, u, y) = generateDataSet("train");

	// train a network: 3 inputs, 
	// the upper subnet has 30 hidden neurons and the lower 20 hidden neurons.
	NARMAL2Network net{ 3, {40}, {30}, "ReLU" };
	int batchSize = 500;
	int nIterations = 3000;
	auto stats = net.trainBatch(X, u, y, 1e-5, batchSize, nIterations, "momentum", 
		{ {"learningRate", 1e-2}, {"gamma", 0.9}, {"decayRate", 0.98} });
	auto yp = net.predict(X, y);

	// get the test set and make predictions
	MatrixXd Xtest;
	VectorXd utest, ytest;
	std::tie(Xtest, utest, ytest) = generateDataSet("test");
	auto ytestp = net.predict(Xtest, utest);
	
	// write results CSV files for further analysis using for example MATLAB
	writeCSV("trainingLoss.csv", stats.trainLossHistory);
	Eigen::MatrixXd testResults{ytest.rows(), 2 };
	testResults.col(0) = ytest;
	testResults.col(1) = ytestp;
	writeCSV("testResults.csv", testResults);
	Eigen::MatrixXd trainingResults{ yp.rows(), 2 };
	trainingResults.col(0) = y;
	trainingResults.col(1) = yp;
	writeCSV("trainingResults.csv", trainingResults);

	std::getchar();
}
```

Results are 

![tl](/doc/img/trainingloss.png)

![](.\doc\img\trainingset.png)

![](.\doc\img\testset.png)

## Reference

Yang, Yue, Cheng Xiang, Shuhua Gao, and Tong Heng Lee. "Data‐driven identification and control of nonlinear systems using multiple NARMA‐L2 models." International Journal of Robust and Nonlinear Control 28, no. 12 (2018): 3806-3833.