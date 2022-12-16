import React from 'react';
import '../../shared/pages/authpage.css'

const Create = () => {
    return (
    <div className="Auth-form-container">
      <form className="Auth-form">
        <div className="Auth-form-content">
          <h3 className="Auth-form-title">Create Data</h3>
          <div className="form-group mt-3">
            <label htmlFor="email">Title</label>
            <input
              type="email"
              className="form-control mt-1"
              placeholder="Enter email"
              id='email'
              name='email'
            />
          </div>
          <div className="form-group mt-3">
            <label htmlFor="password">New</label>
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
              className="btn btn-primary">
              Submit
            </button>
          </div>
        </div>
      </form>
    </div>
    )
}

export default Create;