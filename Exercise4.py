totalsum = []
productList = [10,14,22,33,44,13,22,55,66,77]

def main ():
    print("Supermarket")
    print("===========")
    operating = True
    while operating:
        selection = input("Please select product (1-10) 0 to Quit: ")
        if int(selection) >= 1 and int(selection) <= 10: 
            selectProduct(int(selection))
        elif selection == "0":
            print("Total: ", sum(totalsum))
            amountPaid = input("Payment: ")
            print("Change: ", int(amountPaid) - sum(totalsum))
            break
        else:
            print("Not in range")
        
def selectProduct(productNumber):
    print("Product: ", productNumber, " Price: ", productList[int(productNumber)-1])
    totalsum.append(productList[int(productNumber)-1])
    
main()