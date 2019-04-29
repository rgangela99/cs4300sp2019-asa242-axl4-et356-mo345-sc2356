from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
from app.irsystem.models.search import mediumSearch
from app.irsystem.models.search import youtubeSearch
from app.irsystem.models.search import vid_url_to_title
from app.irsystem.models.search import art_url_to_title

project_name = "MediaFlip"
net_id = "Anjelika Lynne Amog (asa242), Angela Liu (axl4), Emily Tentarelli (et356), Michelle O'Bryan (mo345), Sourabh Chakraborty (sc2356)"

@irsystem.route('/', methods=['GET'])
def search():
	#query_article = request.args.get('input_article_url')
	#query_video = request.args.get('input_video_url')
	query_type = request.args.get('input_type')
	query_url = request.args.get('input_url')
	query_keywords = request.args.get('input_keywords')
	
	if not query_url:
		data = []
		output_message = ""
	elif query_url and ("medium" in query_url) and query_type == "article":
		output_message = "Videos similar to: " + art_url_to_title(query_url)
		data = youtubeSearch(query_url,query_keywords)	
	elif query_url and ("youtube" in query_url) and query_type == "video":
		output_message = "Articles similar to: " + vid_url_to_title(query_url)
		data = mediumSearch(query_url,query_keywords)
	
	elif ("youtube" not in query_url) and query_type == "video":
		data = []
		output_message = "Please enter a valid Youtube link"

	elif query_url and ("medium" not in query_url) and query_type == "article":
		data = []
		output_message = "Please enter a valid Medium link"
	else:
		data = []
		output_message = "Please enter a valid link"
	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)
