import icalendar
from icalendar import Calendar, Event
import pytz
from datetime import datetime, timedelta

def findSundayBaseTime(firstStartTime):
    sundayBaseTime = firstStartTime - \
        timedelta(days=((firstStartTime.isoweekday()) % 7))
    sundayBaseTime = sundayBaseTime.replace(hour = 0, minute = 0, second = 0)
    return sundayBaseTime

def hourDiff(laterTime, firstTime):
    return (laterTime-firstTime).total_seconds()/3600

g = open("/Users/lucaborletti/Desktop/Hack@CMU/F21_schedule.ics", "rb")
gcal = Calendar.from_ical(g.read())

print(gcal)

#find first block class or event

firstStartTime = gcal.walk("VEVENT")[0]["DTSTART"].dt

sundayBaseTime = findSundayBaseTime(firstStartTime)



#now loop through events and add time intervals to list
unsortedStartEndTimes = []

for event in gcal.walk("VEVENT"):
    startTime = hourDiff(event["DTSTART"].dt, sundayBaseTime)
    endTime = hourDiff(event["DTSTART"].dt, sundayBaseTime)
    if 




#dict: initialize

    #x = dict()
    #or ……… x = {}
    
#now initialize value to something

    #X["asdf"] = 3

    #"asdf" index, and storing 3 in that index


temp = str(gcal.walk("VEVENT")[2]["RRULE"]["BYDAY"])

print(temp)

temp = temp.strip("][")

print(temp)

temp = temp.replace("'", "")

print(temp)

temp = temp.split(",")

SU,MO,TU,WE,TH,FR,SA = 7,1,2,3,4,5,6

for i in temp[1:]:

print(temp)

# temp = temp.strip("][")

# print(temp[(temp.find("BYDAY")+8):-2])

# print((temp[(temp.find("BYDAY")+8):-2]).strip("][").strip("\'\'").split(','))

# gcal.walk("VEVENT")[1]["DTSTART"].dt   
    
    #find bydays
    daysRepeat = gcal.walk("VEVENT")[2]["RRULE"]["BYDAY"]
    
    #turn bydays into a list
    # daysRepeat = daysRepeat.strip()

    # if str(event["RRULE"])


print(unsortedStartEndTimes)

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
