import React, {useContext} from 'react'
import { NavLink, useNavigate } from 'react-router-dom';
import './navbar.css'
import { AuthContext } from '../context/auth-context'

const NavLinks = () => {
    const auth = useContext(AuthContext)
    const navigate = useNavigate();
    const SubmitHandler = (e) => {
      e.preventDefault()
      navigate('/Home')
    }
    return (
    <ul className="nav-links">
      <li>
        <NavLink to="/home" exact>
          HOME
        </NavLink>
      </li>
      {auth.isLoggedIn && !(auth.userId == "admin") && (
        <li>
          <NavLink to={`/${auth.userId}/places`}>MY PLACES</NavLink>
        </li>
      )}
      {auth.isLoggedIn && !(auth.userId == "admin") && (
        <li>
          <NavLink to="/places/new">ADD PLACE</NavLink>
        </li>
      )}
      {!auth.isLoggedIn && !(auth.userId == "admin") && (
        <li>
          <NavLink to="/auth">AUTHENTICATE</NavLink>
        </li>
      )}
      

      {auth.isLoggedIn && (auth.userId == "admin") && (
        <li>
          <NavLink to="/admin">ADMIN</NavLink>
        </li>
      )}

      {auth.isLoggedIn && (auth.userId == "admin") && (
        <li>
          <NavLink to="/userlist">USER LIST</NavLink>
        </li>
      )}

    {auth.isLoggedIn && (auth.userId == "admin") && (
        <li>
          <NavLink to="/create">ADMIN CREATE</NavLink>
        </li>
      )}

    {auth.isLoggedIn && (auth.userId == "admin") && (
        <li>
          <NavLink to="/update">ADMIN UPDATE</NavLink>
        </li>
      )}

      {auth.isLoggedIn && (
        <li>
          <button onClick={(event) => { SubmitHandler(event); auth.logout(); }} id="LOGOUT">LOGOUT</button>
        </li>
      )}  
    </ul>
    )

}

export default NavLinks