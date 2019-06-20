/* $(window).on('load', function(event) {
	$('body').removeClass('preloading');
	// $('.load').delay(1000).fadeOut('fast');
	$('.loader').delay(1000).fadeOut('fast');
});
*/

$(document).ready(function(){
	$('#btnsearch').on('click', function (){
		$('#loaderid').addClass('loader');
	});
	
	$('#files').on('change', function (){
		$('#loaderid').addClass('loader');
	});
});

