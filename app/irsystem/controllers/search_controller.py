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
	query_article = request.args.get('input_article_url')
	query_video = request.args.get('input_video_url')
	if not query_article and not query_video:
		data = []
		output_message = ''
	elif query_article:
		output_message = "Videos similar to: " + art_url_to_title(query_article)
		data = youtubeSearch(query_article)
	else:
		output_message = "Articles similar to: " + vid_url_to_title(query_video)
		data = mediumSearch(query_video)
	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)
