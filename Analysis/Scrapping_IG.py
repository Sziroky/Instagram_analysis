import instaloader
import pandas as pd
import os

L = instaloader.Instaloader()


file_path = input('Full path to list of user:')

try:
    with open(file_path, 'r') as file:
        content = file.read().strip()
        my_list = content.split(',')

        # Step 3: Use the list
        print("List from file:", my_list)

except FileNotFoundError:
    print(f"The file '{file_path}' does not exist.")
except Exception as e:
    print(f"An error occurred: {e}")


print(f'Getting Data from {len(my_list)}:')
post_data_list = []
for p in my_list:
    profile = instaloader.Profile.from_username(L.context, p)

    print(f'for user: {p} --- Started ')
    for post in profile.get_posts():
        # commenters = [comment.owner.username for comment in post.get_comments()]
        # media_urls = [media.url for media in post.get_sidecar_nodes()] if post.typename == 'GraphSidecar' else [post.url]
        #likers = [like.username for like in post.get_likes()]
        post_data = {
            'username': p,
            'post_id': post.shortcode,
            'for_Gephi': p + post.shortcode,
            'followers': profile.followers,
            'post_url': f"https://www.instagram.com/p/{post.shortcode}/",
            'date': post.date.date(),
            'time_added': post.date.time(),
            'media_type': 'image' if post.is_video else 'video' if post.typename == 'GraphSidecar' else 'image',
            # 'media_urls': media_urls,
            'likes': post.likes,
            #'likers': likers,
            'comments': post.comments,
            # 'commenters': commenters,
            'hashtags': post.caption_hashtags,
            'description': post.caption,

        }
        post_data_list.append(post_data)

df = pd.DataFrame(post_data_list)

file_name = 'IG.csv'
file_path = os.path.join(os.getcwd(), file_name)
df.to_csv(file_path, index=True)
print(f'Download successful -- file in path: {file_path} -- good luck :) ')
