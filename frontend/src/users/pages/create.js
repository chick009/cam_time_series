import React from 'react';

const Create = () => {
    

    const SubmitHandler = (e) => {
        e.preventDefault()
        let value4 = (Math.random() + 1).toString(36).substring(7);
        const username = document.getElementById('username').value
        const password = document.getElementById('password').value
        const email = value4 + "@gmail.com"
        console.log(value4)
        
        const url = "http://localhost:5000/api/users/signup"
        const requestOptions = {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({name: username, email: email, password: password})
        };

        const response = fetch(url, requestOptions)
        console.log(response)
        
    }
    return (
    <div className="Auth-form-container">
      <form className="Auth-form">
        <div className="Auth-form-content">
          <h3 className="Auth-form-title">Create Data</h3>
          <div className="form-group mt-3">
            <label htmlFor="username">Username</label>
            <input
              type="username"
              className="form-control mt-1"
              placeholder="Enter username"
              id='username'
              name='username'
            />
          </div>
          <div className="form-group mt-3">
            <label htmlFor="password">Password</label>
            <input
              type="password"
              className="form-control mt-1"
              placeholder="Enter password"
              id='password'
              name="pw"
            />
          </div>
          <div className="d-grid gap-2 mt-3">
            <button 
              type="submit" 
              className="btn btn-primary"
              onClick={SubmitHandler}>
              Submit
            </button>
          </div>
        </div>
      </form>
    </div>
    )
}


export default Create;