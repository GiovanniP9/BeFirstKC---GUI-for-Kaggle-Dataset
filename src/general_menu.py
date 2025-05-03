import inspect


class Menu:
    def __init__(self, obj):
        self.obj = obj
        self.methods = self.get_public_methods()

    def get_public_methods(self):
        """Get all public methods of the class (private ones are excluded '__')"""
        return {
            name: method for name, method in inspect.getmembers(self.obj, predicate=inspect.ismethod)
            if not name.startswith('_')
        }

    def show_menu(self):
        """Show the menu of avaible methods"""
        print("\nOperations:")
        for idx, method_name in enumerate(self.methods.keys(), 1):
            print(f"{idx}. {method_name}")
        print(f"0. Exit")

    def select_and_execute(self):
        """Manages the user input and calls the selected methods"""
        while True:
            self.show_menu()
            try:
                choice = int(input("\nSelect an available choice: "))
            except ValueError:
                print("Inserisci un numero valido.")
                continue

            if choice == 0:
                print("Uscita dal menu.")
                break
            elif 1 <= choice <= len(self.methods):
                method_name = list(self.methods.keys())[choice - 1]
                method = self.methods[method_name]

                # Manage inputs
                sig = inspect.signature(method)
                if len(sig.parameters) == 0:
                    # Method without parameters
                    result = method()
                else:
                    # Method with parameters
                    args = []
                    for param in sig.parameters.values():
                        user_input = input(f"Inserisci valore per '{param.name}' ({param.annotation if param.annotation != param.empty else 'str'}): ")
                        args.append(self.cast_input(user_input, param.annotation))
                    result = method(*args)

                if result is not None:
                    print(f"Risultato: {result}")
            else:
                print("Scelta non valida. Riprova.")

    def cast_input(self, value, annotation):
        """Converts the user input into the right type."""
        if annotation == int:
            return int(value)
        elif annotation == float:
            return float(value)
        elif annotation == bool:
            return value.lower() in ['true', '1', 'yes']
        else:
            return value

"""
========== USE CASE ==========

# Create Menu
menu = Menu(calc)

# Execute Menu
menu.select_and_execute()

"""