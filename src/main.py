import os

while True:
    try:
        # Prompt user to select an option
        while True:
            choice = input(f'Select Number: \n1. Train Model \n2. Apply Model\n')
            if choice in ['1', '2']:
                break
            print("Invalid choice. Please enter '1' or '2'.")

        if choice == '1':
            # Prompt user to select a model type
            while True:
                modelType = input('Select Model Type: \n1. Feedforward Neural Network \n2. Recurrent Neural Network\n')
                if modelType == '1':
                    modelIdentifier = 'FNN'
                    break
                elif modelType == '2':
                    modelIdentifier = 'RNN'
                    break
                print("Invalid model type. Please enter '1' or '2'.")

            # Prompt user to enter the number of epochs
            while True:
                epochs = input('Enter number of epochs: ')
                if epochs.isdigit() and int(epochs) > 0:
                    epochs = int(epochs)
                    break
                print("Invalid number of epochs. Please enter a positive integer.")

            # Prompt user to decide whether to plot
            while True:
                toPlot = input('Do you want to plot? (y/n): ').lower()
                if toPlot in ['y', 'n']:
                    plot = toPlot == 'y'
                    break
                print("Invalid input for plotting. Please enter 'y' or 'n'.")

            # Import and train the model
            from TrainModel import TrainModel
            TrainModel(modelIdentifier, epochs, plot)
            break

        elif choice == '2':
            # Display available models
            print('Which model do you want to apply?')
            allModels = os.listdir('src/output/models')
            if not allModels:
                raise FileNotFoundError("No models found in 'src/output/models'. Please train a model first.")
            for i, model in enumerate(allModels):
                print(f'{i + 1}. {model}')

            # Prompt user to select a model
            while True:
                modelChoice = input('Enter model number: ')
                if modelChoice.isdigit() and (1 <= int(modelChoice) <= len(allModels)):
                    modelName = allModels[int(modelChoice) - 1]
                    print(f'You selected {modelName}')
                    break
                print("Invalid model number. Please select a valid number from the list.")

            # Display available MIDI files
            print('Which MIDI file do you want to apply the model to?')
            allMidi = os.listdir('src/input')
            if not allMidi:
                raise FileNotFoundError("No MIDI files found in 'src/input'. Please add a MIDI file to the directory.")
            for i, midi in enumerate(allMidi):
                print(f'{i + 1}. {midi}')

            # Prompt user to select a MIDI file
            while True:
                midiChoice = input('Enter MIDI number: ')
                if midiChoice.isdigit() and (1 <= int(midiChoice) <= len(allMidi)):
                    midiName = allMidi[int(midiChoice) - 1]
                    print(f'You selected {midiName}')
                    break
                print("Invalid MIDI number. Please select a valid number from the list.")

            # Import and apply the model
            from ApplyModel import ApplyModel
            ApplyModel(modelName, midiName)
            break

    except FileNotFoundError as fe:
        print(f"Error: {fe}. Please ensure the required files are available.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}. Please try again.")