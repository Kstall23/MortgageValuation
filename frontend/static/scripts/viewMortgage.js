document.addEventListener("DOMContentLoaded", function(event) {
                let questionIcon = document.getElementById("question");
                let additionalInfo = document.getElementById("explore");
                questionIcon.onclick = function(){
                    additionalInfo.style.display="block";
                }
                let flag_lines = document.getElementsByClassName("minor_line");
                for(let i=0; i<3; i++){
                    let checkIcon = flag_lines[i].previousElementSibling;
                    if(flag_lines[i].textContent.includes("True")){
                        checkIcon.src="/static/icons/check.png";
                        flag_lines[i].textContent = "hello";
                    }
                    else{
                        checkIcon.src="/static/icons/x-mark.png";
                        flag_lines[i].textContent = "hello";
                    }
                }
            });
