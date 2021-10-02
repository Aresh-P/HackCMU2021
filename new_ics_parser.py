from prompt import createStudentGroup
from student_info import StudentInfo
import random
import os, sys
import icalendar
from icalendar import Calendar, Event
import pytz
from datetime import datetime, timedelta

def findSundayBaseTime(firstStartTime):
    ####find the sunday before the week specified in the calendar
    sundayBaseTime = firstStartTime - \
        timedelta(days=((firstStartTime.isoweekday()) % 7))
    sundayBaseTime = sundayBaseTime.replace(hour = 0, minute = 0, second = 0)
    return sundayBaseTime

def hourDiff(laterTime, firstTime):
    ####find hour difference between two times
    return (laterTime-firstTime).total_seconds()/3600

def repeatingEvent(event):
    ####check if has rrule and then if more than 1 "bydays" or repeating days
    if "RRULE" not in str(event):
        return False
    repeatingDays = event["RRULE"]["BYDAY"]
    return len(repeatingDays) > 1

def createStudentBlock(path):
    ####create a list containing names of all ics files
    icsFileNames = os.listdir(path)

    blocks = []

    for icsFileName in (icsFileNames):
        ####open and read ics file
        icsFile = open(f"{path}/{icsFileName}", "rb")
        
        icsCal = Calendar.from_ical(icsFile.read())

        ####check if SIO file or not by analyzing traces of SIO formatting
        if "Instructor" in str(icsCal.walk("VEVENT")[0]):
            firstStartTime = icsCal.walk("VEVENT")[0]["DTSTART"].dt

            sundayStartTime = findSundayBaseTime(firstStartTime)

            sundayEndTime = sundayStartTime + timedelta(days = 7)
        else:
            today = datetime.today()
            
            sundayStartTime = findSundayBaseTime(today)

            sundayEndTime = sundayStartTime + timedelta(days = 7)

        ####create days of week for bi,tri,…-weekly events
        daysOfWeek = {}
        days = ["SU","MO","TU","WE","TH","FR","SA"]
        numbers = [0,1,2,3,4,5,6]
        for i in range(7):
            daysOfWeek[days[i]] = numbers[i]

        ####now LOOP OVER EVENTS IN ICS FILE and add time intervals to list
        for event in icsCal.walk("VEVENT"):
            ####BLOCKS####

            ####initialize datetime variables
            startTime = event["DTSTART"].dt
            endTime = event["DTEND"].dt
            
            ####check if valid event
            if startTime < sundayStartTime or endTime > sundayEndTime:
                continue

            ####measure hours since start of week (saturday midnight)
            hourStartTime = hourDiff(startTime, sundayStartTime)
            hourEndTime = hourDiff(endTime, sundayStartTime)

            ####create an empty dictionary; initialize values to strings
            intervalDict = {}
            intervalDict["start"] = hourStartTime
            intervalDict["stop"] = hourEndTime
            
            ####add dictionaries to blocks
            blocks.append(intervalDict)
            
            # courseName = str(event["DESCRIPTION"])
            # print(f"Main event: {courseName}")
            # print(intervalDict)
            # print("\n")

            if (repeatingEvent(event)):
                repeatingDays = event["RRULE"]["BYDAY"]
                for day in repeatingDays[1:]:
                    dayDifference = daysOfWeek[day] - daysOfWeek[repeatingDays[0]]
                    
                    intervalDict = {}
                    intervalDict["start"] = hourStartTime + 24*dayDifference
                    intervalDict["stop"] = hourEndTime +  24*dayDifference
                    blocks.append(intervalDict)
                    
                    # print(intervalDict)
                    # print("\n")

        # print(blocks)

        #### implement sleep schedule 
        # input("") ***
        for day in range(7):
            intervalDict = {}
            intervalDict["start"] = 24*day
            intervalDict["stop"] =  7 + 24*day
            blocks.append(intervalDict)


        ####return list of dictionaries
        return blocks


def fromSio(icsFile):
    ####check if SIO file or not by analyzing traces of SIO formatting
    icsCal = Calendar.from_ical(icsFile.read())

    if "Instructor" in str(icsCal.walk("VEVENT")[0]):
        return True
    
    return False

def createStudentGoals(icsFile):
    ####generate student goals dictionary where course nums are keys to "goals"
    icsCal = Calendar.from_ical(icsFile.read())
    
    goals = {}

    for event in icsCal.walk("VEVENT"):
        
        rawCourseName = event["DESCRIPTION"]

        courseNum = [i for i in rawCourseName.split() if i.isdigit()][0]

        ####implement goals for each course number
        # input("") ***
        goal = 5*random.random()

        goals[courseNum] = goal

    return goals


def createStudentInfos(path):
    ####create dictionary of student info class objects

    studentInfos = {}

    names = os.listdir(path)

    for name in names:
        
        block = createStudentBlock(f"{path}/{name}")
        
        for icsPath in os.listdir(f"{path}/{name}"): 
            
            icsFile = open(f"{path}/{name}/{icsPath}", "rb")

            if fromSio(icsFile):

                goals = createStudentGoals(icsFile)

        group = createStudentGroup()
        
        studentInfos[name] = studentInfo(
            block,
            goals,
            group
        )

    return studentInfos


# print(os.listdir(""))

print(createStudentInfos("/Users/lucaborletti/Desktop/Hack@CMU/names"))


    
        





# sampleBlocks = [
#     {"start": 12, "stop": 13}, # 12:00 PM - 1:00 PM Sun
#     {"start": 42, "stop": 44}, #  6:00 PM - 8:00 PM Mon
#     # possibly more entries...
# ]

# sampleGoals = {
#     "15151": 6, # Hours/week
#     "15122": 5,
#     # possibly more entries...
# }

# sampleGroup = {"mingroup": 2, "maxgroup": 10} # Just one per student

# sampleInfo = StudentInfo(
#     sampleBlocks,
#     sampleGoals,
#     sampleGroup
#     )


################################################################
################################################################
################################################################
################################################################
################################################################


# icsFile = open("/Users/lucaborletti/Desktop/Hack@CMU/F21_schedule.ics", "rb")

# icsCal = Calendar.from_ical(icsFile.read())

# ####check if SIO file or not by analyzing traces of SIO formatting
# if "Instructor" in str(icsCal.walk("VEVENT")[0]):
#     firstStartTime = icsCal.walk("VEVENT")[0]["DTSTART"].dt

#     sundayStartTime = findSundayBaseTime(firstStartTime)

#     sundayEndTime = sundayStartTime + timedelta(days = 7)
# else:
#     today = datetime.today()
    
#     sundayStartTime = findSundayBaseTime(today)

#     sundayEndTime = sundayStartTime + timedelta(days = 7)

# ####create days of week for bi,tri,…-weekly events
# daysOfWeek = {}
# days = ["SU","MO","TU","WE","TH","FR","SA"]
# numbers = [0,1,2,3,4,5,6]
# for i in range(7):
#     daysOfWeek[days[i]] = numbers[i]

# ####now LOOP OVER EVENTS IN ICS FILE and add time intervals to list
# blocks = []
# for event in icsCal.walk("VEVENT"):
#     ####BLOCKS####

#     ####initialize datetime variables
#     startTime = event["DTSTART"].dt
#     endTime = event["DTEND"].dt
    
#     ####check if valid event
#     if startTime < sundayStartTime or endTime > sundayEndTime:
#         continue

#     ####measure hours since start of week (saturday midnight)
#     hourStartTime = hourDiff(startTime, sundayStartTime)
#     hourEndTime = hourDiff(endTime, sundayStartTime)

#     ####create an empty dictionary; initialize values to strings
#     intervalDict = {}
#     intervalDict["start"] = hourStartTime
#     intervalDict["stop"] = hourEndTime
    
#     ####add dictionaries to blocks
#     blocks.append(intervalDict)
    
#     courseName = str(event["DESCRIPTION"])

#     print(f"Main event: {courseName}")
#     print(intervalDict)
#     print("\n")

#     if (repeatingEvent(event)):
#         repeatingDays = event["RRULE"]["BYDAY"]
#         for day in repeatingDays[1:]:
#             dayDifference = daysOfWeek[day] - daysOfWeek[repeatingDays[0]]
            
#             intervalDict = {}
#             intervalDict["start"] = hourStartTime + 24*dayDifference
#             intervalDict["stop"] = hourEndTime +  24*dayDifference
#             blocks.append(intervalDict)
#             print(intervalDict)
#             print("\n")


# for day in range(7):
#     intervalDict = {}
#     intervalDict["start"] = 24*day
#     intervalDict["stop"] =  7 + 24*day
#     blocks.append(intervalDict)

# print(blocks)






                                    # class StudentInfo:
                                    #     def __init__(self, blocks, goals, group):
                                    #         self.blocks = blocks
                                    #         self.goals = goals
                                    #         self.group = group

                                    # sampleBlocks = [
                                    #     {"start": 12, "stop": 13}, # 12:00 PM - 1:00 PM Sun
                                    #     {"start": 42, "stop": 44}, #  6:00 PM - 8:00 PM Mon
                                    #     # possibly more entries...
                                    # ]

                                    # sampleGoals = {
                                    #     "15151": 6, # Hours/week
                                    #     "15122": 5,
                                    #     # possibly more entries...
                                    # }

                                    # sampleGroup = {"mingroup": 2, "maxgroup": 10} # Just one per student

                                    # sampleInfo = StudentInfo(
                                    #     sampleBlocks,
                                    #     sampleGoals,
                                    #     sampleGroup
                                    #     )



                                    #dict: initialize

                                        #x = dict()
                                        #or ……… x = {}
                                        
                                    #now initialize value to something

                                        #X["asdf"] = 3

                                        #"asdf" index, and storing 3 in that index





# print(gcal)

# print(gcal.walk("VEVENT")[0]["RRULE"]["BYDAY"])

# g2 = open("/Users/lucaborletti/Desktop/Hack@CMU/lborlett@andrew.cmu.edu.ical/lgborletti@gmail.com.ics",\
#      "rb")
# g2cal = Calendar.from_ical(g2.read())

# print(str(g2cal.walk("VEVENT")[1]))

# daysRepeat = "RRULE" in str(g2cal.walk("VEVENT")[1])

# print(daysRepeat)

####find first block class or event



# temp = str(gcal.walk("VEVENT")[2]["RRULE"]["BYDAY"])

# print(temp)

# temp = temp.strip("][")

# print(temp)

# temp = temp.replace("'", "")

# print(temp)

# temp = temp.split(",")

# SU,MO,TU,WE,TH,FR,SA = 7,1,2,3,4,5,6

# for i in temp[1:]:

# print(temp)





# classTimeTuples = [(, gcal.walk("VEVENT")[0])]
# for event in gcal.walk("VEVENT"):
#     startTime = event["DTSTART"].dt
#     endTime = event["DTEND"].dt

#add last one until then 7 + baseSunday



#     print(event["DTSTART"].dt)
#     print(str(event["RRULE"])) #try indexing this string?
        #then for day in days, create new tuple with first dt
        #PLUS (e.g, wed - mon) or today - initialDay so it's reset for week
        #then sort by start days
    

# sessions = [(lecturize(e), calculate_time(e)) for e in cal.walk('vevent')]
