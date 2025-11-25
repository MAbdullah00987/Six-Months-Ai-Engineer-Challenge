"""
Shape Calculator - Main Program
Complete implementation with interactive mode
"""

from shapes import Circle, Rectangle, Triangle


def main():
    """Main function to create and manage shapes."""
    
    # Dictionary to store shape instances
    shapes_collection = {}
    
    print("=" * 50)
    print("Welcome to the Shape Calculator!")
    print("=" * 50)
    
    # Create various shapes and add them to the dictionary
    print("\nCreating shapes...\n")
    
    # Create circles
    circle1 = Circle(radius=5, color="Red")
    shapes_collection["circle1"] = circle1
    
    circle2 = Circle(radius=7.5, color="Blue")
    shapes_collection["circle2"] = circle2
    
    # Create rectangles
    rect1 = Rectangle(length=10, width=5, color="Green")
    shapes_collection["rect1"] = rect1
    
    rect2 = Rectangle(length=8, width=8, color="Yellow")
    shapes_collection["square"] = rect2  # A square is a special rectangle
    
    # Create triangles
    triangle1 = Triangle(side_a=3, side_b=4, side_c=5, color="Purple")
    shapes_collection["triangle1"] = triangle1
    
    triangle2 = Triangle(side_a=6, side_b=8, side_c=10, color="Orange")
    shapes_collection["triangle2"] = triangle2
    
    # Display all shapes
    print("All Shapes in Collection:")
    print("-" * 50)
    
    for key, shape in shapes_collection.items():
        print(f"\nKey: {key}")
        print(shape.display_info())
    
    # Calculate total area using lambda
    total_area = sum(map(lambda shape: shape.calculate_area(), shapes_collection.values()))
    total_perimeter = sum(map(lambda shape: shape.calculate_perimeter(), shapes_collection.values()))
    
    print("\n" + "=" * 50)
    print("Summary Statistics:")
    print("=" * 50)
    print(f"Total number of shapes: {len(shapes_collection)}")
    print(f"Total combined area: {total_area:.2f}")
    print(f"Total combined perimeter: {total_perimeter:.2f}")
    
    # Filter shapes by color using lambda
    print("\n" + "=" * 50)
    print("Finding all red shapes:")
    print("=" * 50)
    red_shapes = list(filter(lambda item: item[1].color == "Red", shapes_collection.items()))
    for key, shape in red_shapes:
        print(f"{key}: {shape}")
    
    # Interactive menu
    interactive_mode(shapes_collection)


def interactive_mode(shapes_collection):
    """Interactive mode for user to add and view shapes."""
    print("\n" + "=" * 50)
    print("Interactive Mode")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Add a new circle")
        print("2. Add a new rectangle")
        print("3. Add a new triangle")
        print("4. View all shapes")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            try:
                radius = float(input("Enter radius: "))
                color = input("Enter color: ").strip() or "Black"
                key = input("Enter a unique key for this shape: ").strip()
                
                if key in shapes_collection:
                    print(f"Key '{key}' already exists. Please choose a different key.")
                else:
                    shapes_collection[key] = Circle(radius, color)
                    print(f"\n{shapes_collection[key]} added successfully!")
                    print(shapes_collection[key].display_info())
            except ValueError as e:
                print(f"Error: {e}")
        
        elif choice == "2":
            try:
                length = float(input("Enter length: "))
                width = float(input("Enter width: "))
                color = input("Enter color: ").strip() or "Black"
                key = input("Enter a unique key for this shape: ").strip()
                
                if key in shapes_collection:
                    print(f"Key '{key}' already exists. Please choose a different key.")
                else:
                    shapes_collection[key] = Rectangle(length, width, color)
                    print(f"\n{shapes_collection[key]} added successfully!")
                    print(shapes_collection[key].display_info())
            except ValueError as e:
                print(f"Error: {e}")
        
        elif choice == "3":
            try:
                side_a = float(input("Enter side A: "))
                side_b = float(input("Enter side B: "))
                side_c = float(input("Enter side C: "))
                color = input("Enter color: ").strip() or "Black"
                key = input("Enter a unique key for this shape: ").strip()
                
                if key in shapes_collection:
                    print(f"Key '{key}' already exists. Please choose a different key.")
                else:
                    shapes_collection[key] = Triangle(side_a, side_b, side_c, color)
                    print(f"\n{shapes_collection[key]} added successfully!")
                    print(shapes_collection[key].display_info())
            except ValueError as e:
                print(f"Error: {e}")
        
        elif choice == "4":
            print("\n" + "-" * 50)
            print("All Shapes in Collection:")
            print("-" * 50)
            for key, shape in shapes_collection.items():
                print(f"\nKey: {key}")
                print(shape.display_info())
        
        elif choice == "5":
            print("\nThank you for using Shape Calculator!")
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")


if __name__ == "__main__":
    main()