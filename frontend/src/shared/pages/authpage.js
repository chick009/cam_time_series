import React, {useContext} from 'react';
import {redirect, Navigate, useNavigate} from 'react-router-dom';
import './authpage.css';
import { useHttpClient } from '../hook/http-hook'
import { AuthContext } from '../context/auth-context';

const AuthPage = () => {
  
  const { isLoading, error, sendRequest, clearError } = useHttpClient();
  const auth = useContext(AuthContext);
  const navigate = useNavigate();
  
  const SubmitHandler = async (e) => {
    e.preventDefault()
    const url = "http://localhost:5000/api/users/" + document.querySelector('#password').value
    const response = await fetch(url)
    const user = await response.json()
    console.log(user)
    // Check whether it's admin 
    const IsAdmin = user.user.isAdmin
    console.log(IsAdmin)
    if(user && IsAdmin == true){
      auth.login("admin")
      navigate('/admin')
    }
    else {
      console.log(user)
    }

    
    /*
    const responseData = await sendRequest(
      'http://localhost:5000/api/users/login',
      'POST',
      JSON.stringify({
        email: document.querySelector('#email').value,
        password: document.querySelector('#password').value
      }),
      {
        'Content-Type': 'application/json'
      }
    );
    console.log(responseData)
    auth.login(responseData.user.id);
    */
    
    
    console.log(response)
    console.log(user)
    
    
  }
  
  return(
    <div className="Auth-form-container">
      <form className="Auth-form">
        <div className="Auth-form-content" id="add">
          <h3 className="Auth-form-title">Sign In</h3>
          <div className="form-group mt-3">
            <label htmlFor="email">Email address</label>
            <input
              type="email"
              className="form-control mt-1"
              placeholder="Enter email"
              id='email'
              name='email'
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

export default AuthPage;
