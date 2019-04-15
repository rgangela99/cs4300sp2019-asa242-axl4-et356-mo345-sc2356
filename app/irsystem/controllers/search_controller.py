from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
from app.irsystem.models.search import mediumSearch

project_name = "Ilan's Cool Project Template"
net_id = "Ilan Filonenko: if56"

@irsystem.route('/', methods=['GET'])
def search():
	query_article = request.args.get('input_article_url')
	query_video = request.args.get('input_video_url')
	if not query_article and not query_video:
		data = []
		output_message = ''
	elif query_article:
		output_message = "Your search: " + query_article
		data = youtubeSearch(query_article)
	else:
		output_message = "Your search: " + query_video
		data = mediumSearch(query_video)
	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)
