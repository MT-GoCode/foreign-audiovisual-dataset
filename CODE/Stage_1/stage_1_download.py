import pandas as pd
from youtubesearchpython import VideosSearch
import json
import yt_dlp
import pandas as pd

class YoutubeFunctions:
    def get_youtube_data(query, results_limit=5):
        video_search = VideosSearch(query, limit=results_limit)
        results = video_search.result()

        # print("YOUTUBE SEARCH RESPONSE: ")
        # print(video_search)
        return [x['id'] for x in results['result'] ]

        # returns a bunch of ID's

    def get_video_info(url):

        ydl_opts = {
            'skip_download': True,  # Do not download the video
            'force_generic_extractor': True,  # This might be necessary if yt-dlp is unable to download video info directly
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Fetch video information
            info_dict = ydl.extract_info(url, download=False)
            # Since '--print-json' is not directly available, we work with the info_dict returned
            # print(info_dict)
            return info_dict


    def download_video(video_url, desired_name, output_path):
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': output_path + desired_name + '.%(ext)s',
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([video_url])
                print("Video downloaded successfully.")
            except Exception as e:
                print("An error occurred while downloading the video: ", e)

class LocalDownloader:
    
    def __init__(self, **kwargs):
        self.__dict__ = dict(kwargs)
        # expects: query_csv, raw_folder, video_csv, results_per_query
        
        self.video_csv_pd = pd.read_csv(self.video_csv)
        self.query_csv_pd = pd.read_csv(self.query_csv)

        print(self.video_csv_pd)


    def fetch_all_remaining_query(self):
        query_list = self.query_csv_pd[self.query_csv_pd['PROCESSED?'] != 1]['QUERY'].tolist()

        return query_list
    
    def confirm_already_downloaded(self, id):
        return id in self.video_csv_pd['VIDEO ID'].values

    def find_video_ids(self, query_list, limit_videos = 1E3, limit_query = 1E3):

        ID_list = []
        cnt_vids = 0

        for query in query_list[:min(limit_query, len(query_list))]:
            to_add = set(YoutubeFunctions.get_youtube_data(query, results_limit = self.results_per_query))
            
            remove_these = set([x for x in to_add if self.confirm_already_downloaded(x)])
            to_add -= remove_these
            
            cnt_vids += len(to_add)

            if cnt_vids >= limit_videos:
                to_add = list(to_add)[:len(to_add) - (cnt_vids - limit_videos)]
                ID_list.append( [query, to_add])
                break

            ID_list.append([query, to_add])

        print("COLLECTED VIDEOS: ")
        print(ID_list)
        return ID_list

    def execute(self):

        query_list = self.fetch_all_remaining_query()
        ID_list = self.find_video_ids(query_list, self.limit_videos, self.limit_query)

        for query, ids in ID_list:
            for id in ids:
                url = "https://www.youtube.com/watch?v=" + id

                try:
                    info = YoutubeFunctions.get_video_info(url)

                    # NOT (good resolution and not english)
                    if not (info.get('width', 0) >= 720 and info.get('height', 0) >= 720 and info.get('language', 'N/A') != 'en' and info.get('duration', 0) < 3600):
                        continue;
                except Exception as e:
                    print("SYSTEM: error fetching data for initial video check: " + id)
                    print(e)
                    continue

                print("VIDEO PASSES.")
                # print(info)
                print("DURATION: " + str(info.get('duration', '0')))
                try:
                    YoutubeFunctions.download_video(url, id, self.raw_folder)
                    print("SYSTEM: successful download of " + id)
                except Exception as e:
                    print("SYSTEM: error downloading video" + id)
                    print(e)
                    continue

                record = self.produce_video_record(url)

                self.update_video_csv(record)
            print("SYSTEM: done processing all videos for query " + query)
            self.update_query_csv(query)
            
    def update_query_csv(self, query):
        self.query_csv_pd.loc[self.query_csv_pd['QUERY'] == query, 'PROCESSED?'] = 1

        # self.query_csv is the PATH
        self.query_csv_pd.to_csv(self.query_csv, index=False) 

    def update_video_csv(self, row):
        self.video_csv_pd.loc[len(self.video_csv_pd)] = row  # adding a row

        # self.video_csv is the PATH
        self.video_csv_pd.to_csv(self.video_csv, index=False)

    def produce_video_record(self, url):

        # try except to account for video being copyrighted
        try:
            info = YoutubeFunctions.get_video_info(url)
            record = pd.Series([info.get('id'), info.get('title', 'N/A'), url, f"{info.get('width', 'N/A')}x{info.get('height', 'N/A')}", info.get('duration', 'N/A'), info.get('language', 'N/A')],
                index = ['VIDEO ID', 'Name', 'Video Link', 'Resolution', 'Duration', 'Language'])
            return record
        except Exception as e:
            print("Could not get video url: ", e)
            return pd.Series(['N/A', 'N/A', url, 'N/A', 'N/A', 'N/A'], index=['VIDEO ID', 'Name', 'Video Link', 'Resolution', 'Duration', 'Language'])
        
# print(YoutubeFunctions.get_video_info("https://www.youtube.com/watch?v=muh361B8lYs")['duration'])