class StudentInfo:
    def __init__(self, blocks, goals, group):
        self.blocks = blocks
        self.goals = goals
        self.group = group

sampleBlocks = [
    {"start": 12, "stop": 13}, # 12:00 PM - 1:00 PM Sun
    {"start": 42, "stop": 44}, #  6:00 PM - 8:00 PM Mon
    # possibly more entries...
]

sampleGoals = {
    "15151": 6, # Hours/week
    "15122": 5,
    # possibly more entries...
}

sampleGroup = {"mingroup": 2, "maxgroup": 10} # Just one per student

sampleInfo = StudentInfo(
    sampleBlocks,
    sampleGoals,
    sampleGroup
    )
