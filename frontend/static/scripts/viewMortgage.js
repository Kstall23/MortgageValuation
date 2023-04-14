document.addEventListener("DOMContentLoaded", function(event) {
                let ppp = document.getElementById("pppt");
                let question = document.getElementById("question");
                let explore = document.getElementById("explore");
                let lines = document.getElementsByClassName("minor_line")
                for(let i=0; i<lines.length; i++){
                    let check = lines[i].previousElementSibling;
                    if(lines[i].textContent.includes("False")){
                        check.src="/static/icons/check.png"
                    }
                    else{
                        check.src="/static/icons/x-mark.png"
                        ppp.id = "pppf"
                        question.src = "/static/icons/bq.png"
                    }
                }
                question.onclick = function(){
                    explore.style.display="block";
                }
            });