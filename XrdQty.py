#"/usr/bin/python
# -*- coding: utf-8 -*-

"""XrdQty.py: Python module for creation of synthetic x-ray diffraction pattern of minerals for training a CNN, which can be used to quantify mineral compounds"""

__author__ = "Markus Schindler"
__copyright__ = "Copyright 2025"

__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Markus Schindler"
__email__ = "schindlerdrmarkus@gmail.com"
__status__ = "Development"

# Built-in / Generic Imports
import os, random, sys, time
from math import cos, exp, pi
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Calculate X-ray diffraction peak using Cauchy-Distribution Function
class XRDCauchy:
    def __init__(self, material):
        maxCrystalliteSize_nm = 500
        self.solidPhase = material
        self.CrystalliteSize = random.random() * maxCrystalliteSize_nm
    
    # FWHM = Full Width at Half Maximum
    def FWHM(self, braggAngle):
        # Wavelength: Cu K-Alpha radiation
        shapeFactor, wavelength_nm = 0.94, 0.1541838
        return 180 / pi * (shapeFactor * wavelength_nm) / (self.CrystalliteSize * cos(braggAngle * pi / 180))
    
    def Cauchy(self, twoTheta):
        sum = 0
        for itervar in self.solidPhase:
            sum += itervar[1] * self.FWHM(0.5 * itervar[0]) / pi * 0.5 / ((twoTheta - itervar[0])**2 + self.FWHM(0.5 * itervar[0])**2)
        return sum
        
# Calculate X-ray diffraction pattern based on mineral composition
class PDFData:
    def __init__(self, start_angle, stop_angle, angle_steps):
        # Definition of measurement / simulation range
        self.startAngle = start_angle
        self.stopAngle = stop_angle
        self.resolution = angle_steps
        
        # Procedure to collect all data files with the structure folder
        # Material structural data as peak location at two Theta and corresponding intensity
        # Data is required to calculate Probability Density Functions
        data_dir = "structure"
        
        # Lists to store the reflexes and intensities and mineral phase name
        self.phases = []
        self.minerals = []

        # Recursively scan for .csv files
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(".csv"):
                    file_path = os.path.join(root, file)
                # Load the two-column data (2 Theta, Intensities) into a Numpy Array
                array = np.loadtxt(file_path, skiprows = 1, delimiter = ",")
                # Use the base file name without extensions as the variable name
                variable_name = os.path.splitext(file)[0]
                safe_name = variable_name.replace("-", "_").replace(" ","_")
                if variable_name.isidentifier():
                    self.phases.append(array)
                    self.minerals.append(variable_name)
                else:
                    print(f'Skipping invalid variable name: {variable_name}')

        # Load the maximum intensity data from file
        file_path = "maxIntensity.csv"
        df = pd.read_csv(file_path, sep = ",")
        df.index = "max. Intensity", "Fraction"
        if df.shape[1] is not len(self.minerals):
            print(f'Maximum intensity data file differs from content in the structure folder!')
        self.max_intensity = np.zeros((len(self.minerals), 1), dtype = float)
        self.fraction = np.zeros((len(self.minerals), 1), dtype = float)
        i = 0
        for row in df:
            self.max_intensity[i] = df[row]["max. Intensity"]
            self.fraction[i] = df[row]["Fraction"]
            i += 1
        
        # Other properties for the X-ray diffraction measurements
        self.numberOfPhases = len(self.minerals)
        self.angle = np.zeros(self.resolution, dtype = float)
        self.intensity = np.zeros(self.resolution, dtype = float)
        for itervar in range(self.resolution):
            self.angle = np.linspace(self.startAngle, self.stopAngle, self.resolution)

    def composition(self):
        # Creating random mineral composition for creating training data
        phaseFractions = np.zeros((len(self.minerals)), dtype = float)
        phase_variation = (1 - self.fraction.max()) / len(self.minerals)
        for itervar in range(len(self.minerals)):
            phaseFractions[itervar] = self.fraction[itervar, 0] + random.random() * phase_variation
        phaseFractions /= float(phaseFractions.sum())
        return phaseFractions

    def clearance(self):
        self.intensity = np.zeros((self.resolution), dtype = float)
    
    def singlePhasePattern(self, phase, level):
        self.clearance()
        anyPattern = XRDCauchy(phase)
        for itervar in range(self.resolution):
            self.intensity[itervar] += level * anyPattern.Cauchy(self.angle[itervar])

    def noiseFunction(self, twoTheta, maximum):
        # Setting maximum to zero has then no impact
        location, broadness = 27, 100
        return maximum * exp(-1 * (location - twoTheta)**2 / broadness)
    
    def backgroundNoise(self):
        maximum = random.uniform(1, 5)
        for itervar in range(self.resolution):
            self.intensity[itervar] += self.noiseFunction(self.angle[itervar], maximum)

    def multiPhasePattern(self):
        self.clearance()
        materialLevels = self.composition()
        for itervar in range(len(self.minerals)):
            self.singlePhasePattern(self.phases[itervar], materialLevels[itervar] * self.max_intensity[itervar, 0] / 100)
        self.backgroundNoise()
        return materialLevels, self.intensity

# Definition the CNN Model
class CNN(nn.Module):
    def __init__(self, input_length, output_size):
        super(CNN, self).__init__()
        # First convolutional layer with 32 filters
        self.conv1 = nn.Conv1d(in_channels = 1,
                               out_channels = 32,
                               kernel_size = 3,
                               stride = 1,
                               padding = 1)
        # First pooling layer with kernel size 2
        self.pool1 = nn.MaxPool1d(kernel_size = 2, stride = 2)
        # Second convolutional layer with 64 filters
        self.conv2 = nn.Conv1d(in_channels = 32,
                               out_channels = 64,
                               kernel_size = 3,
                               stride = 1,
                               padding = 1)
        # Second pooling layer with kernel size 2
        self.pool2 = nn.MaxPool1d(kernel_size = 2, stride = 2)
        # Third convolutional layer with 64 filters
        self.conv3 = nn.Conv1d(in_channels = 64,
                               out_channels = 128,
                               kernel_size = 3,
                               stride = 1,
                               padding = 1)
        # Compute the flattened size after conv3
        flattened_dim = 128 * (input_length // 4)
        # Fully connected layers
        self.fc1 = nn.Linear(flattened_dim, 1024)
        self.fc2 = nn.Linear(1024, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1) # -> (batch_size, 1, input_length)
        x = self.pool1(self.relu(self.conv1(x))) # -> (batch_size, 32, input_length / 2)
        x = self.pool2(self.relu(self.conv2(x))) # -> (batch_size, 64, input_length / 2)
        x = self.relu(self.conv3(x)) # -> (batch_size, 64, input_length / 2)
        x = x.view(x.size(0), -1) # flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class XrdQty:
    def __init__(self, start_angle, stop_angle, angle_steps, model_name = 'new_model'):
        self.startAngle = start_angle
        self.stopAngle = stop_angle
        self.resolution = angle_steps
        self.model_name = model_name
        # initialize the pattern once per instance
        self.xrayPattern = PDFData(self.startAngle, self.stopAngle, self.resolution)
        self.materialLevels, self.intensity = self.xrayPattern.multiPhasePattern()
        self.angle = self.xrayPattern.angle
        self.numberOfPhases = len(self.materialLevels)

    def progressBar(countedValue, total, suffix=''):
        barLength = 100
        filledUpLength = int(round(barLength * countedValue / float(total)))
        percentage = round(100.0 * countedValue / float(total), 1)
        bar = "=" * filledUpLength + "-" * (barLength - filledUpLength)
        sys.stdout.write(f"[{bar}] {percentage}% ...{suffix}\r")
        sys.stdout.flush()

    def synthetic(self, runs):
        inputData = np.zeros((runs, len(self.intensity)), dtype=float)
        outputData = np.zeros((runs, self.numberOfPhases), dtype=float)
        for itervar in range(runs):
            # recreate the pattern for each run if desired
            xrayPattern = PDFData(self.startAngle, self.stopAngle, self.resolution)
            mat, inten = xrayPattern.multiPhasePattern()
            outputData[itervar, :] = mat
            inputData[itervar, :] = inten
            self.__class__.progressBar(itervar + 1, runs)
        return inputData, outputData

    def create_training_data(self):
        featureData, labelData = self.synthetic(500)
        np.savetxt("feature.csv", featureData, delimiter = ",", header = "Intensities")
        np.savetxt("label.csv", labelData, delimiter = ",", header = "Fraction")

    # Define model training
    def train(self, num_epochs = 100, criterion = None, optimizer = None):
        if criterion is None or optimizer is None:
            raise ValueError("Criterion and Optimizer must be provided!")
        for epoch in range(num_epochs):
            outputs = self.model(self.X_train)
            if torch.cuda.is_available:
                outputs = outputs.cuda()
            loss = criterion(outputs, self.y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 5 == 0:
                print(f'Epoch[{epoch + 1}/{num_epochs}], Loss:{loss.item():.4f}')

    # Function for returning the index of a maximum value in a 1-D numpy array
    def max_index(self, arr):
        if arr.size == 0:
            raise ValueError("Cannot find maximum index of an empty array")
        return np.argmax(arr)
    
    # Function for calculation of mean absolute error
    def mae(self, values, prediction):
        _ = PDFData(self.startAngle, self.stopAngle, self.resolution)
        fraction = _.fraction
        phase = self.max_index(fraction)
        sum = 0
        for itervar in range(len(prediction)):
            sum += abs(prediction[itervar][phase] - values[itervar][phase])
        return float(sum / len(prediction))

    def model_training(self):
        # Load the generated XRD data from CSV files
        X = np.loadtxt("feature.csv", delimiter = ",", skiprows = 1)
        y = np.loadtxt("label.csv", delimiter = ",", skiprows = 1)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

        # Standardize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

        # Convert data to PyTorch tensors
        self.X_train = torch.from_numpy(X_train).float()
        self.X_test = torch.from_numpy(X_test).float()
        self.y_train = torch.from_numpy(y_train).float()
        self.y_test = torch.from_numpy(y_test).float()

        # GPU Handling
        if torch.cuda.is_available:
            self.X_train = self.X_train.cuda()
            self.X_test = self.X_test.cuda()
            self.y_train = self.y_train.cuda()
            self.y_test = self.y_test.cuda()

        # Create an instance of the CNN model
        input_size = self.X_train.shape[1] # Number of features in the XRD pattern
        output_size = self.y_train.shape[1] # Number of mineral phases
        self.model = CNN(input_size, output_size)
        print(self.model)
        print("Input  Size =", input_size, "\nOutput Size =", output_size)

        # Define loss function and optimizer
        criterion = nn.MSELoss() # Mean Standard Error Loss
        optimizer = optim.SGD(self.model.parameters(), lr = 0.01)

        if torch.cuda.is_available:
            self.model = self.model.cuda()

        # Train the model
        self.train(criterion = criterion, optimizer = optimizer)

        # Evaluate models performance
        outputs = self.model(self.X_test)
        mae_xrd = self.mae(self.y_test, outputs)
        print(f'Mean Absolute Error: {mae_xrd:.4f}')

        # Save the trained model
        filepath = self.model_name + ".pth"
        torch.save(self.model.state_dict(), filepath) 
        print("Model saved to", filepath)

    def load_model(self):
        input_size = self.resolution # Number of features in the XRD pattern
        output_size = self.numberOfPhases # Number of mineral phases
        self.model = CNN(input_size, output_size)
        filepath = self.model_name + ".pth"
        if torch.cuda.is_available:
            self.model.load_state_dict(torch.load(filepath, weights_only = True))
            self.model.cuda()
        else:
            self.model.load_state_dict(torch.load(filepath, weights_only = True, map_location=torch.device('cpu')))
        self.model.eval()
        print("Input  Size =", input_size, "\nOutput Size =", output_size)
        print("Model loaded from", filepath)

    def predict(self, xrd_pattern):
        # Loading the base data for prediction
        X_predict = np.loadtxt(xrd_pattern, delimiter = ",", skiprows = 1)
        
        # Standardize the input pattern
        scaler = StandardScaler()
        xrd_pattern_scaled = scaler.fit_transform(X_predict)

        # Convert to PyTorch tensor
        xrd_tensor = torch.from_numpy(xrd_pattern_scaled).float()

        if torch.cuda.is_available:
            xrd_tensor = xrd_tensor.cuda()

        # Predict using the model
        with torch.no_grad():
            prediction = self.model(xrd_tensor)
        return prediction

# Example usage
if __name__ == "__main__":
    xrd_qty = XrdQty(start_angle=10, stop_angle=90, angle_steps=8501)
    xrd_qty.create_training_data()
    xrd_qty.model_training()
    xrd_qty.load_model()

    # Create a synthetic pattern for prediction
    _, test_pattern = xrd_qty.xrayPattern.multiPhasePattern()
    predicted_composition = xrd_qty.predict(test_pattern)

    print("Predicted Composition:", predicted_composition)
