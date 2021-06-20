import PersonDetector as p

personDector = p.PersonDetector()

for i in range(0, 10):
    personDector.setImagePathToRead("time_square_01.png")
    personDector.execute()
    personDector.draw()
