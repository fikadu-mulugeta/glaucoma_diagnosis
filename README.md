#Glaucoma Diagnosis Using Deep Learning and Machine Learning Models
This project focuses on the diagnosis of glaucoma using retinal images, employing various machine learning and deep learning techniques. The aim is to develop an accurate and generalized model to assist in early glaucoma detection.
Overview
The project implements and compares the following models:
•	Support Vector Machine (SVM)
•	K-Nearest Neighbors (KNN)
•	Convolutional Neural Network (CNN) (custom-built)
•	Transfer learning using pre-trained models:
		AlexNet
		GoogLeNet
		SqueezeNet
Datasets
The data was obtained from five publicly available sources:
1.	ORIGA Dataset
	Source: Online Retinal Fundus Image Dataset for Glaucoma Analysis and Research (ORIGA).
	Description: 650 images (168 glaucomatous, 482 non-glaucoma) from the Singapore Malay Eye Study (SiMES).
	Format: JPG, annotated by experts.
2.	ACRIMA Dataset
	Source: FISABIO Oftalmología Médica in Valencia, Spain.
	Description: 705 images (396 glaucomatous, 309 normal), annotated by glaucoma experts.
	Size: 1054x1054, 96dpi resolution, 24-bit depth.
3.	LAG Dataset
	Source: Kaggle (Glaucoma Dataset).
	Description: 4854 images (1711 glaucomatous, 3143 normal).
	Size: 500x500, 96dpi resolution, 24-bit depth.
4.	REFUGE Dataset
	Source: REFUGE challenge dataset.
	Description: 1200 images equally split between glaucomatous and normal categories.
	Size: 1054x1054, 96dpi resolution, 24-bit depth.
5.	HRF Dataset
	Source: High-Resolution Fundus Image Database, Friedrich-Alexander-Universität Erlangen-Nürnberg, Germany.
	Description: 30 images (15 glaucomatous, 15 normal), annotated by experts.
	Size: 2336x3504, 72dpi resolution, 24-bit depth.
A data named mydatanew.xslx is extracted from these dataset and used to train the machine learning models. The deep learning models required no feature extraction.
The primary proposed model is a custom-built CNN trained from scratch using publicly available datasets from five sources, with one source(LAG) used for training and the remaining for generalization.
Project Files
•	cnn.m: Implements a custom CNN architecture for glaucoma detection.
•	SVM.m: Trains an SVM classifier using polynomial kernels for glaucoma classification.
•	knn.m: Trains a KNN classifier with Hamming distance for classification.
•	TransferAlexnet.m: Adapts the AlexNet architecture for glaucoma detection using transfer learning.
•	TransferGooglenet.m: Fine-tunes the GoogLeNet architecture for glaucoma detection.
•	TransferSqueezenet.m: Fine-tunes the SqueezeNet architecture for glaucoma detection.
Results
The machine learning models were evaluated using metrics such as:
•	Accuracy
•	Precision
•	Sensitivity
•	Specificity
•	Confusion Matrix
Deep Learnng models were evaluated using
•	ROC Curve and AUC
How to Use
1.	Clone this repository:
git clone https://github.com/your-username/Glaucoma-Diagnosis.git
2.	Ensure MATLAB R2021a or later is installed.
3.	Place datasets in the specified directories as per the scripts.
4.	Run individual scripts for specific models:
		cnn.m: Custom CNN model.
		SVM.m: SVM model.
		knn.m: KNN model.
		TransferAlexnet.m, TransferGooglenet.m, TransferSqueezenet.m: Transfer learning models.
