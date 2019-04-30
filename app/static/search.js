//collapsible result display boxes
/*function create_result_boxes(){
	var result_expand_buttons = document.getElementsByClassName("expand_result_button");
	var result_idx;
	for (result_idx = 0; result_idx < result_expand_buttons.length; result_idx++){
		result_expand_buttons[result_idx].addEventListener("click", function(){
			//this.classList.toggle("expanded");
			var info = this.nextElementSibling;
			if (info.style.display === "block"){
				info.style.display = "none"
			} else {
				info.style.display = "block"
			}
		});
	}
}*/

function toggle_info_box(button_element){
	var info = button_element.nextElementSibling;
	if (info.style.display === "block"){
		info.style.display = "none"
	} else {
		info.style.display = "block"
	}
}
