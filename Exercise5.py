def main():
    sentence = str(input("Please enter sentence:"))
    print(my_join(my_split(sentence, ' '),','))
    print(my_join(my_split(sentence, ' '),'\n'))

def my_join(splitSentence, separator):
    
    joinedString = ''
    for c in splitSentence:
        joinedString += c + separator
    return joinedString[:-1]


def my_split(sentence, separator):
    split_value = []
    tmp = ''
    for c in sentence:
        if c == separator:
            split_value.append(tmp)
            tmp = ''
        else:
            tmp += c
    #adds the last "word" to the list
    if tmp:
       split_value.append(tmp)
   
    return split_value

main()
