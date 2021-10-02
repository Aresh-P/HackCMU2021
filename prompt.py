def maxInput(studentName):
    max = input(f'{studentName}, What is the maximum people you want to work with? --> ')
    if max.isnumeric():
        max = int(max)
    valid = False
    if isinstance(max,int)==False:
        while valid == False:
            max = input(f'Please input a valid number of max people {studentName} --> ')
            if max.isnumeric():
                max = int(max)
            if max < 0:
                continue
            else:
                valid = True
    return max

def minInput(studentName):
    min = input(f'{studentName}, What is the minimum people you want to work with? --> ')
    if min.isnumeric():
        min = int(min)
    valid = False
    if isinstance(min,int)==False:
        while valid == False:
            min = input(f'Please input a valid number of max people {studentName} --> ')
            if min.isnumeric():
                min = int(min)
            if min < 0:
                continue
            else:
                valid = True
    return min
def createStudentGroup(studentName):
    studentInput = {}
    max = maxInput(studentName)
    min = minInput(studentName)
    studentInput["mingroup"] = min
    studentInput["maxgroup"] = max
    return studentInput
