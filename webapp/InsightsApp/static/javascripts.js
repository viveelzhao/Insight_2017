$(document).ready(function(){
    $("#cancel").click(function(){
        $("#intro").val("");
    });
})



readFromServer = function(url){
    var xhttp = new XMLHttpRequest();
    var text
    xhttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            return this.responseText
        }
    };
    xhttp.open('GET', url, false);
    xhttp.send();
    text = xhttp.onreadystatechange()
    return text
}



$(document).ready(function(){
    $("#main-tab1").click(function(){
        var text = readFromServer("/output_1")
        $("#pane1").html("<p>"+ text + "</p>");
    })
})

$(document).ready(function(){
    $("#main-tab2").click(function(){
        var text = readFromServer("/output_2")
        $("#pane2").html("<p>"+ text + "</p>");
    })
})

$(document).ready(function(){
    $("#main-tab3").click(function(){
        var text = readFromServer("/output_3")
        $("#pane3").html("<p>"+ text + "</p>");
    })
})

$(document).ready(function(){
    $("#main-tab4").click(function(){
        var text = readFromServer("/output_4")
        $("#pane4").html("<p>"+ text + "</p>");
    })
})