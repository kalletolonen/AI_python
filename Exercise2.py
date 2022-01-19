from json.tool import main
from re import I

itemList = []
incorrect = "Incorrect selection."

def main():
    operating = True
    while operating:
        print("Would you like to")
        print("(1)Add or")
        print("(2)Remove items or")
        userInput = input("(3)Quit?: ")
        
        if userInput == "3":
            print("The following items remain in the list:")
            for i in itemList:
                print(i)
            break
        elif userInput == "1":
            itemList.append(addItem())
        elif userInput == "2":
            print("There are ", len(itemList), " items in the list.")
            itemNumber = input("Which item is deleted?:")
            if(checkInput(itemNumber)):
                del itemList[int(itemNumber)]
            else:
                print(incorrect)                
        else:
            print(incorrect)

def addItem():
    addItemName = input("What will be added?: " )
    return addItemName

def deleteItem(itemNumber):
    itemList.remove(itemNumber)

def checkInput(input):
    try:
        val = int(input)
        if (val >= 0 and val <= len(itemList) - 1):
            return True
    except ValueError:
            print(incorrect)



main()