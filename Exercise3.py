def main():
    a = []
    number = int(input("Please enter the Total number of Elements : "))
    for i in range(number):
        value = int(input("Please enter the %d Element : " %i))
        a.append(value)
    print ("The sorted List in Ascending Order : ", bubble_sort(a))
    print("main")

def bubble_sort(a):
    arrayLenght = len(a)

    # Traverse through all array elements
    for i in range(arrayLenght-1):
    # range(arrayLenght) also work but outer loop will
    # repeat one time more than needed.
 
        # Last i elements are already in place
        for j in range(0, arrayLenght-i-1):
 
            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if a[j] > a[j + 1] :
                a[j], a[j + 1] = a[j + 1], a[j]
    return a
    
main()