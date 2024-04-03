import re

def extract_speed_from_filename(file_name):
    """
    Extracts the speed from the filename.
    Returns the numeric speed for fixed speeds, or -1 for variable speeds.
    """
    fixed_speed_match = re.search(r"PGB_(\d+)_", file_name)
    if fixed_speed_match:
        return int(fixed_speed_match.group(1))
    variable_speed_match = re.search(r"Variable_speed", file_name)
    if variable_speed_match:
        return -1  # Special value for variable speeds
    return None