import math
import random
import icalendar
import copy
from csv_ical import Convert
import time
import pandas as pd
from icalendar import Calendar, Event
from copy import deepcopy
from cost import cost
import numpy as np
import torch
from torch import nn, optim
from math import ceil, floor
from allocate_groups import *
from groups_to_output import *
from module_approach_parse_ics import *
from new_ics_parser import *
from online_allocate_groups import *
from optimizer import *
from preliminary_ics_parser import *
from prompt import *
from student_info import *
from student import *

def main():
    print("Hello!")
    print("Welcome to the CMU Study Group Finder")
    print("Our goal is to match you with other CMU students with similar study group preferences as you in order to help you find a study group")
    print()
    input("Please Press any key when you are ready to begin")
    print()
    studentFolder = input("Please input the name of your study folder ----> ")
    studentInfo = createStudentInfos(studentFolder)
    student = AllStudents(studentInfo)

