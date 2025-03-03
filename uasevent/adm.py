import json
import math
import numpy as np
import xml.dom.minidom
import xml.etree.ElementTree as ET


class TrajectoryToADMXML():

    def __init__(self,
                 json_trajectory_file,
                 event_name="Event",
                 ):
        
        self.json_trajectory_file = json_trajectory_file
        self.event_name = event_name
        self._block_id = 1
        self._rtime = 0.0

    def write_adm(self, filepath):
        self.element_tree_setup()
        self.segment_process()
        self.element_tree_close()
        self.write_xml(filepath)
        self._rtime = 0.0
        self._block_id = 1

    @property
    def json_trajectory_file(self):
        return self._json_trajectory_file

    @json_trajectory_file.setter
    def json_trajectory_file(self, filepath):
        self.trajectory = json.load(open(filepath))
        self.total_duration = sum(
            self.estimate_duration(segment["start"],
                                   segment["end"],
                                   segment["speeds"])
            for segment in self.trajectory.values())
        self._json_trajectory_file = filepath

    def element_tree_setup(self):
        """Constructs boilerplate and root objects needed for ADM XML file."""
        # Create the root element
        self._ebuCoreMain = ET.Element("ebuCoreMain", {
            "xmlns:dc": "http://purl.org/dc/elements/1.1/",
            "xmlns": "urn:ebu:metadata-schema:ebuCore_2014",
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "schema": "EBU_CORE_20140201.xsd",
            "xml:lang": "en"
        })

        # Create the self._coreMetadata element
        self._coreMetadata = ET.SubElement(self._ebuCoreMain,
                                           "coreMetadata")

        # Create the format element
        format_elem = ET.SubElement(self._coreMetadata, "format")

        # Create the self._audioFormatExtended element
        self._audioFormatExtended = ET.SubElement(
            format_elem, "audioFormatExtended", 
            {"version": "ITU-R_BS.2076-2"})

        # Create the self._audioProgramme element
        self._audioProgramme = ET.SubElement(
            self._audioFormatExtended, "audioProgramme", {
                "audioProgrammeID": "APR_1001",
                "audioProgrammeName": "Default"
        })
        ET.SubElement(
            self._audioProgramme, "audioContentIDRef").text = "ACO_1001"

        # Create the self._audioContent element
        self._audioContent = ET.SubElement(
            self._audioFormatExtended, "audioContent", {
                "audioContentID": "ACO_1001",
                "audioContentName": "Default"
        })
        ET.SubElement(self._audioContent, "audioObjectIDRef").text = "AO_1001"

        # Create the self._audioObject element
        self._audioObject = ET.SubElement(
            self._audioFormatExtended, "audioObject", {
                "audioObjectID": "AO_1001",
                "audioObjectName": "Event",
                "start": "00:00:00.00000",
                "duration": f"{self.decimal_to_hms(self.total_duration)}"
        })
        ET.SubElement(self._audioObject, "audioPackFormatIDRef"
                      ).text = "AP_00031001"
        ET.SubElement(self._audioObject, "audioTrackUIDRef"
                      ).text = "ATU_00000001"

        # Create the self._audioPackFormat element
        self._audioPackFormat = ET.SubElement(
            self._audioFormatExtended, "audioPackFormat", {
                "audioPackFormatID": "AP_00031001",
                "audioPackFormatName": self.event_name,
                "typeLabel": "0003",
                "typeDefinition": "Objects"
        })
        ET.SubElement(self._audioPackFormat, 
                      "audioChannelFormatIDRef").text = "AC_00031001"

        # Create root XML element with audioObjectID
        self._audioChannelFormat = ET.SubElement(
            self._audioFormatExtended, "audioChannelFormat",
            audioChannelFormatID="AC00031001",
            audioChannelFormatName="Event",
            typeLabel="0003",
            typeDefinition="Objects")

    def create_audio_blocks_segment(self, segment):
        start, end, speeds = segment["start"], segment["end"], segment["speeds"]
        total_duration = self.estimate_duration(start, end, speeds)

        # Check if acceleration is present (speed changes)
        if speeds[0] != speeds[1]:
            # Interpolate only when accelerating or decelerating
            steps = max(int(total_duration * 10), 1)  # Use 10 steps per second
            step_duration = total_duration / steps
            
            for i, (azimuth, elevation, distance) in enumerate(
                self.interpolate_positions(start, end, steps)):
                if i:
                    block = self.create_audio_block_format(
                        azimuth, elevation, distance, 
                        self._rtime, step_duration, self._block_id)
                    self._audioChannelFormat.append(block)
                    self._block_id += 1
                    self._rtime += step_duration
        else:
            # Single block pair for constant-speed segments
            if not self._rtime:
                azimuth, elevation, distance = self.cartesian_to_polar(*start)
                block = self.create_audio_block_format(
                    azimuth, elevation, distance, 
                    self._rtime, 0.0, self._block_id)
                self._audioChannelFormat.append(block)

            azimuth, elevation, distance = self.cartesian_to_polar(*end)
            block = self.create_audio_block_format(
                azimuth, elevation, distance,
                self._rtime, total_duration, self._block_id)
            self._audioChannelFormat.append(block)

            self._rtime += total_duration
            self._block_id += 1
        
    def element_tree_close(self):
        audioTrackUID = ET.SubElement(self._audioFormatExtended,
                                      "audioTrackUID", UID="ATU_00000001")
        ET.SubElement(audioTrackUID, "audioChannelFormatIDRef"
                      ).text = "AC_00031001"
        ET.SubElement(audioTrackUID, "audioPackFormatIDRef"
                      ).text = "AP_00031001"
    
    def write_xml(self, filepath):
        # Convert XML tree to string
        adm_xml = ET.tostring(self._ebuCoreMain,
                              encoding="unicode", method="xml")

        # Pretty-print the XML with indentation
        dom = xml.dom.minidom.parseString(adm_xml)
        pretty_xml = dom.toprettyxml(indent="   ")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(pretty_xml)

    def segment_process(self):
        for _, segment in self.trajectory.items():
            self.create_audio_blocks_segment(segment)

    def calculate_distance(self, start, end):
        """Calculate Euclidean distance between two points in Cartesian space"""
        return math.sqrt(sum((s - e) ** 2 for s, e in zip(start, end)))

    def estimate_duration(self, start, end, speeds):
        """Estimate duration using average speed."""
        distance = self.calculate_distance(start, end)
        avg_speed = sum(speeds) / len(speeds)  # Average speed over the segment
        return distance / avg_speed if avg_speed > 0 else 0
    
    def decimal_to_hms(self, decimal_seconds):
        # Extract hours, minutes, and seconds
        hours = int(decimal_seconds // 3600)
        minutes = int((decimal_seconds % 3600) // 60)
        seconds = decimal_seconds % 60

        # Format the output with 2 decimal places for the seconds
        return f"{hours:02}:{minutes:02}:{seconds:011.8f}"
    
    def interpolate_positions(self, start, end, steps):
        """Interpolate azimuth, elevation, and distance between two points."""
        start_az, start_el, start_dist = self.cartesian_to_polar(*start)
        end_az, end_el, end_dist = self.cartesian_to_polar(*end)
        
        # Linear interpolation in polar coordinates
        azimuths = np.linspace(start_az, end_az, steps)
        elevations = np.linspace(start_el, end_el, steps)
        distances = np.linspace(start_dist, end_dist, steps)
        
        return zip(azimuths, elevations, distances)
    
    def create_audio_block_format(self, azimuth, elevation, distance, 
                                  rtime, duration, block_id):
        """Create an audioBlockFormat XML element."""
        block = ET.Element("audioBlockFormat", 
                           audioBlockFormatID=f"AB_{block_id:08d}", 
                           rtime=f"{self.decimal_to_hms(rtime)}", 
                           duration=f"{self.decimal_to_hms(duration)}")
        # ET.SubElement(block, "rtime").text = "0"  # Relative time (start at 0)
        # ET.SubElement(block, "duration").text = f"{duration:.2f}"
        
        # Polar position (Azimuth, Elevation, Distance)
        ET.SubElement(block, "position", coordinate="azimuth"
                      ).text = f"{azimuth:.2f}"
        ET.SubElement(block, "position", coordinate="elevation"
                      ).text = f"{elevation:.2f}"
        ET.SubElement(block, "position", coordinate="distance"
                      ).text = f"{distance:.2f}"
        
        ET.SubElement(block, "gain").text = "1.0"

        return block
    
    def cartesian_to_polar(self, x, y, z):
        """Convert Cartesian (X, Y, Z) to Polar (Azi, Elev, Radius)."""
        distance = math.sqrt(x**2 + y**2 + z**2)
        azimuth = math.degrees(math.atan2(y, x)) % 360  # wrap @ 360
        elevation = math.degrees(math.atan2(z, math.sqrt(x**2 + y**2)))

        return azimuth, elevation, distance