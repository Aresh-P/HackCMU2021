import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sin, cos
from math import ceil, floor
from student_info import StudentInfo

cycles = torch.Tensor(
    [7*24, 24, 8, 4, 2, 1]) # From a week to an hour
numCycles = len(cycles)
inSize = 2*numCycles # Neural network inputs include sin and cos
period = max(cycles) # Hours in a week
hourRes = 2 # 30-minute intervals
scheduleLen = period*hourRes # Number of intervals
hiddenSizes = [16]
times = (torch.range(scheduleLen)+0.5)/hourRes # Center of each interval

# Accepts time in hours (0 to 7*24)
# Returns tensor of inputs to neural network
# (trig functions of varying frequency)
def timeToNNInput(t):
    angles = 2*np.pi*t/cycles
    cosines = torch.cos(angles)
    sines = torch.sin(angles)
    return torch.cat((cosines, sines), 0)

# Accepts list of class/other blocks,
# where each block is a dict containing bounds in hours
# Returns bool array indicating which intervals are free
def blocksToFreeTime(blocks):
    freeTime = torch.BoolTensor(torch.ones((scheduleLen,)))
    for block in blocks:
        start = ceil(block["start"]*hourRes)
        stop = floor(block["stop"]*hourRes)
        for interval in range(start, stop):
            freeTime[interval] = False
    return freeTime

# see student_info.py
class Student:
    def __init__(self, studentInfo):
        self.blocks = studentInfo.blocks
        self.goals = studentInfo.goals
        self.group = studentInfo.group
        self.freeTime = blocksToFreeTime(self.blocks)
        self.courses = list(self.goals.keys())
        self.numCourses = len(self.courses)
        self.studySchedule = StudySchedule(self.numCourses+1)
    def updateSchedule(self, scheduleLen):
        self.scheduleTensor = torch.zeros(scheduleLen, self.numCourses+1)
        for i in range(scheduleLen):
            if self.freeTime[i]:
                self.scheduleTensor[i]=self.studySchedule(times[i])
        self.schedule = {}
        for c in range(self.numCourses+1):
            course = self.courses[c]
            self.schedule[course] = self.scheduleTensor[:, c]
        
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

class AllStudents:
    def __init__(self, students):
        self.students = students # {str -> Student}
        self.roster = dict() # {str -> [str]}
        for name in students:
            student = students[name]
            for course in student.courses:
                if course not in self.roster:
                    self.roster[course] = []
                self.roster[course].append(name)
    def loss(self):
        J = 0
        for course in self.roster: # course name
            directory = self.roster[course] # list of student names
            courseSize = len(directory) # number of students in course
            avgSchedule = torch.zeros(scheduleLen)
            studyError = 0
            for name in directory: # name of given student
                student = directory[name] # Student object
                student.updateSchedule() # create schedule : {str -> FloatTensor by time}
                courseSchedule = student.schedule[course]
                avgSchedule += courseSchedule
                studyTime = torch.sum(courseSchedule)/hourRes
                studyGoal = student.goals[course]
                studyError += ((studyTime-studyGoal)/studyGoal)^2
            avgSchedule /= courseSize
            studyError /= courseSize
            J += torch.linalg.vector_norm(avgSchedule)/scheduleLen
            J -= studyError                
    def fit(self):
        pass