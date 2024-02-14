form =document.getElementById("recommendation-form")

form.addEventListener('submit', (event) => {
    event.preventDefault()
    fetch(form.action, {
        mode: 'cors',
        method: 'post',
        body: new FormData(form),
    }).then((res) => {
        //Change here
        console.log(res)
    })
})