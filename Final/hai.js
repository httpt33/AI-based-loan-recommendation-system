form.addEventListener('submit', (event) => {
    event.preventDefault()
    fetch(form.action, {
        method: 'post',
        body: new FormData(form),
    }).then((res) => {
        if (!res.ok) {
            throw new Error(HTTP error! Status: ${response.status})
        }
        // CHANGE THE PAGE HERE
    })
})