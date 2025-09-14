import os

# Define the directory you want to iterate through
directory = r"whatsup_vlms\data\controlled_images"

# Iterate through all files in the directory
for filename in os.listdir(directory):
    # Check if "police_cruiser" is in the filename
    if "above" in filename:
        if not "_above_" in filename:

            # Create the new name by replacing underscores with spaces
            new_filename = filename.replace("above", "on")
            # Get the full path for both the old and new filenames
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            # Rename the file
            os.rename(old_file, new_file)
            print(f"Renamed: {old_file} -> {new_file}")
