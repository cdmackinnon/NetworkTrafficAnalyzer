from Algorithms.Aggregator import main as aggregator_main
from Algorithms.DecisionTree import main as decisiontree_main
from Algorithms.LogisticRegression import main as logisticregression_main
from Algorithms.SVM import main as svm_main

if __name__ == "__main__":
    while True:
        print("Type 1 to assemble and preprocess the data set")
        print("Type 2 to build a decision tree")
        print("Type 3 to perform logistic regression")
        print("Type 4 to create a SVM")
        print("Type anything else to exit")
        
        user_input = input("Enter your choice: ")
        print("\n")

        if user_input == '1':
            data_directory = input("Input the collected data directory: ")
            print("Assembling and preprocessing data set...")
            aggregator_main(directory_path = data_directory)

        elif user_input == '2':
            print("Building a decision tree...")
            decisiontree_main()

        elif user_input == '3':
            print("Performing logistic regression...")
            logisticregression_main()

        elif user_input == '4':
            print("Creating a SVM...")
            svm_main()

        else:
            print("Exiting program...")
            break 

