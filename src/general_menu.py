import inspect


class Menu:
    """
    Dynamically generates a menu interface for interacting with public methods of a class.
    
    It uses introspection to display method names and handle user input, 
    including basic type casting for arguments.
    """

    def __init__(self, obj):
        """
        Initialize the menu with an object instance.

        Parameters:
        - obj (object): An instance of the class to be inspected and interacted with.
        """
        self.obj = obj
        self.methods = self.get_public_methods()

    def get_public_methods(self):
        """
        Retrieve all public methods from the object (excluding those starting with underscores).

        Returns:
        - dict[str, method]: Dictionary of method names and method references.
        """
        return {
            name: method for name, method in inspect.getmembers(self.obj, predicate=inspect.ismethod)
            if not name.startswith('_')
        }

    def show_menu(self):
        """
        Display a numbered menu of all available public methods.
        """
        print("\nOperations:")
        for idx, method_name in enumerate(self.methods.keys(), 1):
            print(f"{idx}. {method_name}")
        print(f"0. Exit")

    def select_and_execute(self):
        """
        Handle user input to select and execute methods interactively.
        
        Prompts for method arguments when necessary and attempts to cast them to expected types.
        """
        while True:
            self.show_menu()
            try:
                choice = int(input("\nSelect an available choice: "))
            except ValueError:
                print("Please enter a valid number.")
                continue

            if choice == 0:
                print("Exiting menu.")
                break
            elif 1 <= choice <= len(self.methods):
                method_name = list(self.methods.keys())[choice - 1]
                method = self.methods[method_name]

                sig = inspect.signature(method)

                if len(sig.parameters) == 0:
                    # Method with no parameters
                    result = method()
                else:
                    # Method with parameters - collect user inputs
                    args = []
                    for param in sig.parameters.values():
                        annotation = param.annotation if param.annotation != param.empty else str
                        user_input = input(f"Enter value for '{param.name}' ({annotation.__name__}): ")
                        args.append(self.cast_input(user_input, annotation))
                    result = method(*args)

                if result is not None:
                    print(f"Result: {result}")
            else:
                print("Invalid selection. Try again.")

    def cast_input(self, value, annotation):
        """
        Attempt to cast the user input to the expected type.

        Parameters:
        - value (str): Raw input from the user.
        - annotation (type): Expected type for the input.

        Returns:
        - value converted to the expected type, or original string if type is unknown.
        """
        try:
            if annotation == int:
                return int(value)
            elif annotation == float:
                return float(value)
            elif annotation == bool:
                return value.lower() in ['true', '1', 'yes']
            else:
                return value  # fallback to string
        except Exception as e:
            print(f"Could not cast input '{value}' to {annotation}. Using raw input.")
            return value


"""
========== USE CASE ==========

# Create Menu
menu = Menu(calc)

# Execute Menu
menu.select_and_execute()

"""