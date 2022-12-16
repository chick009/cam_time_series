import logo from './logo.svg';
import { BrowserRouter, Routes, Route, Link, useNavigate} from 'react-router-dom';
import {useState, useCallback} from 'react'
import Home from './home'
import Auth from './shared/pages/authpage'
import Admin from './users/pages/admin'
import Create from './users/pages/create'
import Update from './users/pages/update'
import Users from './users/pages/usersall'
import { AuthContext } from './shared/context/auth-context';
import NavBar from './shared/components/navbar'
import './App.css';
// import Create from './users/pages/create'
const App = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [userId, setUserId] = useState(false);

  const login = useCallback(uid => {
    setIsLoggedIn(true);
    console.log(isLoggedIn)
    setUserId(uid);
  }, []);

  const logout = useCallback(() => {
    setIsLoggedIn(false);
    setUserId(null);
  }, []);

  return (

    <BrowserRouter>
    <AuthContext.Provider
      value={{
        isLoggedIn: isLoggedIn,
        userId: userId,
        login: login,
        logout: logout
      }}
    >
      <NavBar />
      
        
        <hr/>
        <Routes>
          <Route path="/home" element={<Home/>} />
          <Route path="/auth" element={<Auth />} />

          {userId == "admin" && 
            <Route path='/admin' element={<Admin />} />
          }
          {userId == "admin" && 
            <Route path='/update' element={<Update/>} />
          }
          {userId == "admin" && 
            <Route path='/create' element={<Create/>} />
          }
          {userId == "admin" && 
            <Route path='/userlist' element={<Users/>} />
          }
          
        </Routes>
      
    </AuthContext.Provider>
    </BrowserRouter>
  )
}

export default App;
