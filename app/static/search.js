//collapsible result display boxes


function toggle_info_box(button_element){
	var info = button_element.nextElementSibling;
	if (info.style.display === "block"){
		info.style.display = "none"
	} else {
		info.style.display = "block"
	}
}
