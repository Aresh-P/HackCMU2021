import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sin, cos
from math import ceil, floor

# Need to get these from elsewhere
totalCourses = 3
nameToNumber = {'a': 0, 'b': 1, 'c': 2}

cycles = torch.Tensor(
    [7*24, 24, 8, 4, 2, 1]) # From a week to an hour
numCycles = len(cycles)
inSize = 2*numCycles # Neural network inputs include sin and cos
period = cycles[0]
hourRes = 1 # 12 for 5-minute intervals
scheduleLen = period
hiddenSizes = [32, 32]

# Accepts time in hours (0 to 7*24)
# Returns tensor of inputs to neural network
# (trig functions of varying frequency)
def timeToNNInput(t):
    angles = 2*np.pi*t/cycles
    cosines = torch.cos(angles)
    sines = torch.sin(angles)
    return torch.cat((cosines, sines), 0)

# Accepts list of free blocks,
# where each block is a dict containing bounds in hours
# Returns bool array indicating which intervals are free
def blocksToFreeTime(freeBlocks):
    freeTime = torch.BoolTensor(np.zeros(7*24*hourRes))
    for freeBlock in freeBlocks:
        start = ceil(freeBlock["start"]*hourRes)
        stop = floor(freeBlock["stop"]*hourRes)
        for interval in range(start, stop):
            freeTime[interval] = True
    return freeTime

class Student:
    def __init__(self, name, freeTime, courses):
        self.name = name
        self.freeTime = freeTime
        self.courses = courses
        self.numCourses = len(courses)
        self.studySchedule = studySchedule(numCourses+1)
        
# Neural network to model schedule
# Requires number of courses
# Inputs time, outputs probabilities of each course (or nothing)
class StudySchedule(nn.Module):
    def __init__(self, outSize):
        super(StudySchedule, self).__init__()
        self.outSize = outSize
        self.layerSizes = [inSize]+hiddenSizes+[outSize]
        self.numWeights = len(hiddenSizes)+1
        self.fcLayers = nn.Sequential(
            nn.Linear(layerSizes[w], layerSizes[w+1])
            for w in range(numWeights))
        self.softmax = nn.Softmax()
        
    def forward(self, t):
        return self.softmax(self.fcLayers(timeToNNInput(t)))
        
print(timeToNNInput(0))
print(blocksToFreeTime([{"start": 5, "stop": 7}, {"start": 8, "stop": 12}]))
