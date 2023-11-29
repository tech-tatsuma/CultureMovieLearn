from moviepy.editor import VideoFileClip
import os
import pandas as pd
import argparse
from tqdm import tqdm

# Function to split a video into segments, each one second long
def split_video(video_path, output_folder):
    # Load the video file
    clip = VideoFileClip(video_path)
    duration = int(clip.duration)

    outputs = []

    # Split and save the video into one-second segments
    for i in tqdm(range(duration-1), desc="Splitting video"):
        # Set the time range for the video segment
        subclip = clip.subclip(i, i+1)

        # Save the split video to a file (frame rate set to 25fps)
        output_path = os.path.join(output_folder, f"segment_{i}.mp4")

        # Check if the file already exists
        if not os.path.exists(output_path):
            # Save the split video to a file only if it doesn't exist (frame rate set to 25fps)
            subclip.write_videofile(output_path, codec="libx264", fps=25, verbose=False, logger=None)
        else:
            print(f"File {output_path} already exists. Skipping.")

        outputs.append(output_path)

    clip.close()

    return outputs

# Function to create a DataFrame of labeled video segments from paths and CSV content
def create_labeled_video_csv(video_paths, csv_content, duration):
    # Parse the CSV content to store time ranges and statuses in a dictionary
    time_ranges = csv_content.split('\n')
    print(time_ranges) # ['57,0', '320,2', '50,1', '30,0', '70,2']
    status_dict = {}
    prev_time = 0

    for range_info in time_ranges:
        if not range_info.strip() or 'during,status' in range_info:
            continue
        time, status = map(int, range_info.split(','))
        status_dict[range(prev_time, prev_time+time)] = status
        prev_time = prev_time+time

    status_dict[range(prev_time, duration)] = 0

    # Assign labels to each video segment
    labeled_data = []
    for video_path in video_paths:
        file_name = os.path.basename(video_path)
        segment_time = int(file_name.split('_')[-1].split('.')[0])
        label = None
        for time_range, status in status_dict.items():
            if segment_time in time_range:
                label = status
                break
        if label is None:
            label = 0
        labeled_data.append((video_path, label))

    # Create and return a DataFrame
    df = pd.DataFrame(labeled_data, columns=['video_path', 'status'])
    return df

# Function to read a CSV file and return its content as a string
def read_csv_to_string(csv_path):
    with open(csv_path, 'r') as file:
        lines = file.readlines()
    content = "\n".join(line.strip() for line in lines[1:])
    return content

# Function to write a DataFrame to a CSV file
def write_dataframe_to_csv(df, output_path):
    # Write the DataFrame to a CSV file
    df.to_csv(output_path, index=False)

# Main function to process the video and timeline CSV, and output labeled data
def main(video_path, timeline_csv_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Split the video and save the segments to the output folder
    video_segments = split_video(video_path, output_folder)

    # Read the timeline data from the CSV
    timeline_data = read_csv_to_string(timeline_csv_path)

    # Create a DataFrame with labeled video data
    labeled_df = create_labeled_video_csv(video_segments, timeline_data, len(video_segments))

    # Get the parent directory of the output folder
    parent_folder = os.path.dirname(output_folder)

    # Convert the first column of the DataFrame (video paths) to relative paths
    labeled_df['video_path'] = labeled_df['video_path'].apply(
        lambda x: os.path.relpath(x, parent_folder)
    )

    # Write the results to a CSV file in the parent directory of the output folder
    output_csv_path = os.path.join(parent_folder, 'labeled_videos.csv')
    write_dataframe_to_csv(labeled_df, output_csv_path)

# Parse command line arguments and call the main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str)
    parser.add_argument('--timeline_csv_path', type=str)
    parser.add_argument('--output_folder', type=str)
    args = parser.parse_args()

    main(args.video_path, args.timeline_csv_path, args.output_folder)