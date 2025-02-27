import json
import math
import numpy as np
import xml.dom.minidom
import xml.etree.ElementTree as ET


def decimal_to_hms(decimal_seconds):
    # Extract hours, minutes, and seconds
    hours = int(decimal_seconds // 3600)
    minutes = int((decimal_seconds % 3600) // 60)
    seconds = decimal_seconds % 60

    # Format the output with 2 decimal places for the seconds
    return f"{hours:02}:{minutes:02}:{seconds:011.8f}"

def cartesian_to_polar(x, y, z):
    """Convert Cartesian (X, Y, Z) to Polar (Azimuth, Elevation, Distance)."""
    distance = math.sqrt(x**2 + y**2 + z**2)
    azimuth = math.degrees(math.atan2(y, x)) % 360  # Convert to degrees, wrap 0-360
    elevation = math.degrees(math.atan2(z, math.sqrt(x**2 + y**2)))  # Convert to degrees
    return azimuth, elevation, distance

def calculate_distance(start, end):
    """Calculate Euclidean distance between two points in Cartesian space."""
    return math.sqrt(sum((s - e) ** 2 for s, e in zip(start, end)))

def estimate_duration(start, end, speeds):
    """Estimate duration using average speed."""
    distance = calculate_distance(start, end)
    avg_speed = sum(speeds) / len(speeds)  # Average speed over the segment
    return distance / avg_speed if avg_speed > 0 else 0

def interpolate_positions(start, end, steps):
    """Interpolate azimuth, elevation, and distance between two points."""
    start_az, start_el, start_dist = cartesian_to_polar(*start)
    end_az, end_el, end_dist = cartesian_to_polar(*end)
    
    # Linear interpolation in polar coordinates
    azimuths = np.linspace(start_az, end_az, steps)
    elevations = np.linspace(start_el, end_el, steps)
    distances = np.linspace(start_dist, end_dist, steps)
    
    return zip(azimuths, elevations, distances)

def create_audio_block_format(azimuth, elevation, distance, rtime, duration, block_id):
    """Create an audioBlockFormat XML element."""
    block = ET.Element("audioBlockFormat", audioBlockFormatID=f"AB_{block_id:08d}", rtime=f"{decimal_to_hms(rtime)}", duration=f"{decimal_to_hms(duration)}")
    # ET.SubElement(block, "rtime").text = "0"  # Relative time (start at 0)
    # ET.SubElement(block, "duration").text = f"{duration:.2f}"
    
    # Polar position (Azimuth, Elevation, Distance)
    ET.SubElement(block, "position", coordinate="azimuth").text = f"{azimuth:.2f}"
    ET.SubElement(block, "position", coordinate="elevation").text = f"{elevation:.2f}"
    ET.SubElement(block, "position", coordinate="distance").text = f"{distance:.2f}"
    
    ET.SubElement(block, "gain").text = "1.0"
    
    return block


### Create Boilerplate ###

# Create the root element
ebuCoreMain = ET.Element("ebuCoreMain", {
    "xmlns:dc": "http://purl.org/dc/elements/1.1/",
    "xmlns": "urn:ebu:metadata-schema:ebuCore_2014",
    "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
    "schema": "EBU_CORE_20140201.xsd",
    "xml:lang": "en"
})

# Create the coreMetadata element
coreMetadata = ET.SubElement(ebuCoreMain, "coreMetadata")

# Create the format element
format_elem = ET.SubElement(coreMetadata, "format")

# Create the audioFormatExtended element
audioFormatExtended = ET.SubElement(format_elem, "audioFormatExtended", {"version": "ITU-R_BS.2076-2"})

# Create the audioProgramme element
audioProgramme = ET.SubElement(audioFormatExtended, "audioProgramme", {
    "audioProgrammeID": "APR_1001",
    "audioProgrammeName": "Default"
})
ET.SubElement(audioProgramme, "audioContentIDRef").text = "ACO_1001"

# Create the audioContent element
audioContent = ET.SubElement(audioFormatExtended, "audioContent", {
    "audioContentID": "ACO_1001",
    "audioContentName": "Default"
})
ET.SubElement(audioContent, "audioObjectIDRef").text = "AO_1001"

# Create the audioObject element
audioObject = ET.SubElement(audioFormatExtended, "audioObject", {
    "audioObjectID": "AO_1001",
    "audioObjectName": "Event",
    "start": "00:00:00.00000",
    "duration": f"{decimal_to_hms(total_duration)}"
})
ET.SubElement(audioObject, "audioPackFormatIDRef").text = "AP_00031001"
ET.SubElement(audioObject, "audioTrackUIDRef").text = "ATU_00000001"

# Create the audioPackFormat element
audioPackFormat = ET.SubElement(audioFormatExtended, "audioPackFormat", {
    "audioPackFormatID": "AP_00031001",
    "audioPackFormatName": "Event",
    "typeLabel": "0003",
    "typeDefinition": "Objects"
})
ET.SubElement(audioPackFormat, "audioChannelFormatIDRef").text = "AC_00031001"


### Create audioBlocks ###

# load trajectory
trajectory = json.load(open('flights/demo_flight.json', 'r'))
# Calculate total duration
total_duration = sum(
    estimate_duration(segment["start"], segment["end"], segment["speeds"]) 
    for segment in trajectory.values())

# Create root XML element with audioObjectID
audioChannelFormat = ET.SubElement(audioFormatExtended, "audioChannelFormat",
    audioChannelFormatID="AC00031001",
    audioChannelFormatName="Event",
    typeLabel="0003",
    typeDefinition="Objects")

# Process each flight segment
block_id = 1
rtime = 0.0
for segment_name, segment in trajectory.items():
    start, end, speeds = segment["start"], segment["end"], segment["speeds"]
    
    total_duration = estimate_duration(start, end, speeds)
    
    # Check if acceleration is present (speed changes)
    if speeds[0] != speeds[1]:
        # Interpolate only when accelerating or decelerating
        steps = max(int(total_duration * 10), 1)  # Use 10 steps per second
        step_duration = total_duration / steps
        
        for i, (azimuth, elevation, distance) in enumerate(
            interpolate_positions(start, end, steps)):
            if i:
                block = create_audio_block_format(
                    azimuth, elevation, distance, 
                    rtime, step_duration, block_id)
                audioChannelFormat.append(block)
                block_id += 1
                rtime += step_duration
    else:
        # Single block pair for constant-speed segments
        if not rtime:
            azimuth, elevation, distance = cartesian_to_polar(*start)
            block = create_audio_block_format(
                azimuth, elevation, distance, rtime, 0.0, block_id)
            audioChannelFormat.append(block)

        azimuth, elevation, distance = cartesian_to_polar(*end)
        block = create_audio_block_format(
            azimuth, elevation, distance, rtime, total_duration, block_id)
        audioChannelFormat.append(block)

        rtime += total_duration

        block_id += 1

# add audioFormat tags
audioTrackUID = ET.SubElement(audioFormatExtended, "audioTrackUID", UID="ATU_00000001")
ET.SubElement(audioTrackUID, "audioChannelFormatIDRef").text = "AC_00031001"
ET.SubElement(audioTrackUID, "audioPackFormatIDRef").text = "AP_00031001"

# Convert XML tree to string
adm_xml = ET.tostring(ebuCoreMain, encoding="unicode", method="xml")
# print(adm_xml)

# Pretty-print the XML with indentation
dom = xml.dom.minidom.parseString(adm_xml)
pretty_xml = dom.toprettyxml(indent="   ")

# Save to file
with open("output.adm.xml", "w", encoding="utf-8") as f:
    f.write(pretty_xml)
