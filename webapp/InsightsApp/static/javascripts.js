//$(document).ready(function(){
//    $("#cancel").click(function(){
//        $("#intro").val("");
//    });
//})

(function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
(i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
})(window,document,'script','https://www.google-analytics.com/analytics.js','ga');

ga('create', 'UA-93094107-1', 'auto');
ga('send', 'pageview');



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