import csv
import chardet
import datetime
import os
import re


def convert_time(secs):
    return datetime.timedelta(seconds=round(secs))
def read_segments(file_path):
    """
    Read the segments from a CSV file and return them as a dictionary.
    """
    print(f" Reading segments from: {file_path}")
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
    print(f"Encoding of {file_path} is {result['encoding']}")

    with open(file_path, 'r', newline='', encoding=result['encoding']) as f:
        reader = csv.reader(f)
        next(reader)  # skip the header row
        segments = []
        for row in reader:
            chunk = {}
            start, end, speaker, text = row
            chunk["start"] = float(start)
            chunk["end"] = float(end)
            chunk["speaker"] = speaker
            chunk["text"] = text
            segments.append(chunk)
    return segments

def read_processed_data(file_path):
    """
    Read the data that has been processed in InqScribe and return it as a dictionary.
    The date structure will have to be converted back using time_to_counter().
    """
    print(f" Reading segments from: {file_path}")
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
    print(f"Encoding of {file_path} is {result['encoding']}")
    with open(file_path, 'r', newline='', encoding=result['encoding']) as f:
        reader = csv.reader(f)
        next(reader)  # skip the header row
        segments = []

        for row in reader:
            try:
                #If by accident there are extra columns, just add them to the text
                start, end, speaker, *additional_values = row
                if validate_timestamp(start) and validate_timestamp(end):
                    text = ' '.join([str(value) for value in additional_values])
                    chunk = {
                        "start": time_to_counter(start),
                        "end": time_to_counter(end),
                        "speaker": speaker,
                        "text": text
                    }
                    segments.append(chunk)
            except ValueError:
                print(f"Skipping invalid row: {row}")

    return segments

def counter_to_time(counter):
    '''
    Convert a counter (in seconds) to a time string. Note that a round() is used, so some precision is lost!
    Example: 1.23456789 becomes [00:00:01.23], which can then be used in InqScribe.
    '''

    hours = int(counter // 3600)
    minutes = int((counter % 3600) // 60)
    seconds = int((counter % 60) // 1)
    milliseconds = int(round((counter % 1) * 100))
    return "[{:02d}:{:02d}:{:02d}.{:02d}]".format(hours, minutes, seconds, milliseconds)

def time_to_counter(time_str):
    '''
    This takes the InqScribe time string and converts it back into a counter (in seconds).
    '''
    time_str = time_str[1:-1]  # Remove the square brackets
    hours, minutes, seconds_ms = time_str.split(":")
    seconds, milliseconds = seconds_ms.split(".")
    return (int(hours) * 3600) + (int(minutes) * 60) + int(seconds) + (int(milliseconds) / 100)

def validate_timestamp(timestamp):
    """
    Check if the timestamp is in the InqScribe format.
    """
    pattern = r"\[\d{2}:\d{2}:\d{2}\.\d{2}\]"
    return re.match(pattern, timestamp) is not None

def prepare_for_inqscribe(segments, file_path):
    """
    Save the segments to a CSV file.

    Use the Text Wizard in Excel when importing!
    Select: Delimiter Commas, Text qualifier Double quotes, File origin 65001: Unicode (UTF-8)
    Tick: "My data has headers"
    Make sure to export as CSV UTF-8 (Comma delimited) (*.csv)

    """
    print(f"Saving segments to: {file_path}")
    if os.path.exists(file_path):
        print (f"File {file_path} already exists.")
        promt = input(f"Press y to overwrite {file_path} or any other key to abort:")
        if promt != "y":
            print(f"{file_path} will not be overwritten.")
            return
    else:
        print("Writing to file...")
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Start', 'End', 'Speaker', 'Text'])
            for i, segment in enumerate(segments):
                segment["start"] = counter_to_time(segment["start"])
                segment["end"] = counter_to_time(segment["end"])
                writer.writerow([segment["start"], segment["end"], segment["speaker"], segment["text"]])

def create_final_transcription(file_path='segments_intermediate.csv', split_by = 'character count'):
    """
    Create the final output from the segments of an audio file, and write them to a CSV file.
    """
    with open(file_path, 'rb') as file:
        character_detect = chardet.detect(file.read())

    # Maximum character count for each block of speaker text
    # 1000 characters is about 150 words
    # 600 characters is about 100 words
    max_character_count = 200

    #Do you want to split by character count or by full stops?

    if split_by == 'character count':
        print(f"Splitting by character count: {max_character_count}")
    elif split_by == 'full stops':
        print("Splitting by full stops.")

    with open(file_path, newline='', encoding=character_detect['encoding']) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        segments = []
        current_segment = None

        for row in reader:
            print(f"Raw Line: {row}")
            #speaker = row['Speaker']
            #start = row['Start']
            #end = row['End']
            #text = row['Text']

            speaker = row['Speaker']
            start = row['Start']
            end = row['End']
            text = row['Text']

            #print the whole thing to find errors
            print(f"{start} {end} {speaker} {text}")

            # If the text ends with a question mark, set the speaker to 'Question'
            # if text.endswith('?'):
            #     speaker = 'Question'

            if current_segment is None:
                # Start a new segment if no current segment exists
                current_segment = {
                    'Start': start,
                    'End': end,
                    'Speaker': speaker,
                    'Text': text
                }
            elif split_by == 'character count' and len(current_segment['Text']) + len(text) <= max_character_count and speaker == current_segment['Speaker']:
                # Add the row to the current segment if it doesn't exceed the character count limit

                current_segment['End'] = end
                current_segment['Text'] += ' ' + text
            elif split_by == 'full stops' and speaker == current_segment['Speaker'] and re.search(r'\,\s*$', text):
                # Add the row to the current segment if it ends with a full stop
                current_segment['End'] = end
                current_segment['Text'] += ' ' + text
            else:
                # Add the current segment to the list and start a new segment
                segments.append(current_segment)
                current_segment = {
                    'Start': start,
                    'End': end,
                    'Speaker': speaker,
                    'Text': text
                }

        # Add the last segment
        if current_segment is not None:
            segments.append(current_segment)

    # To avoid overwriting, check if the file exists and add a counter if it does

    now = datetime.datetime.now()
    formatted_date = now.strftime("%d-%m-%y %H.%M") #Output format: "01-01-21 13.00"
    output_file = f"{file_path} created {formatted_date}.csv"

    if os.path.exists(output_file):
        counter = 1
        while os.path.exists(output_file):
            output_file = f"{file_path} created {formatted_date} ({counter}).csv"
            counter += 1

    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Start', 'End', 'Speaker', 'Text'])
            for i, segment in enumerate(segments):
                segment["Text"] = segment["Text"].lstrip()
                # Check if Start value is not None before accessing its substring
                if segment["Start"] is not None:
                    segment["Start"] = segment["Start"][:-4] + "]"
                if segment["End"] is not None:
                    segment["End"] = segment["End"][:-4] + "]"
                # Next line also removes superfluous whitespace
                writer.writerow([segment["Start"], segment["End"], segment["Speaker"], " ".join(segment["Text"].split())])
                # Or without removing superfluous whitespace
                #writer.writerow([segment["Start"], segment["End"], segment["Speaker"], segment["Text"]])
    except:
        print("Error writing to file.")
    return output_file

# As the first step, read the segments from the CSV file
# These are the segments created by using OpenAI and the speaker diarization
segments = read_segments("Whisper segments.csv")

# This is the second step, where we prepare the segments for InqScribe
# Time stamps are converted to the InqScribe format
# The function takes two arguments: the segments and the name of the output file
prepare_for_inqscribe(segments, "Prepared for InqScribe.csv")

# This is the third step, where the corrected transcription is read from the CSV file
processed = read_processed_data("Corrected segments.csv")

# This is the fourth step, where the final transcription is created
# Speaker diarization is used to group the segments by speaker
final = create_final_transcription("Final transcription.csv")
print(f"Final transscription with filename {final} created successfully!")