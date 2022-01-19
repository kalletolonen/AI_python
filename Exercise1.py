from json.tool import main
    
def main():
    operating = True
    while operating:
        userInput = input("Write something (quit ends): ")
        if userInput == "quit":
            break
        else:
            print(tester(userInput))

def tester(userInput, givenstring = "Too short"):
    if (len(userInput) < 10):
        return givenstring
    else:
        givenstring = userInput
        return givenstring

main()